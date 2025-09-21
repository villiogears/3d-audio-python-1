import pyaudio
import numpy as np
import time
import sys
try:
	import torch
	TORCH_AVAILABLE = True
except Exception:
	torch = None
	TORCH_AVAILABLE = False


# Advanced PyTorch binaural processor with fractional delay (stateful)
class AdvancedBinauralProcessor:
	def __init__(self, rate=44100, max_delay_ms=2.0, device=None):
		if not TORCH_AVAILABLE:
			raise RuntimeError('PyTorch is not available')
		self.rate = rate
		self.max_delay = int(rate * max_delay_ms / 1000)  # samples
		self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
		self.prev = torch.zeros(self.max_delay + 2, dtype=torch.float32, device=self.device)

	def fractional_delay(self, x: torch.Tensor, delay: float) -> torch.Tensor:
		# delay >= 0 (samples, may be fractional)
		n = int(torch.floor(torch.tensor(delay)).item())
		frac = delay - n
		L = x.shape[0]
		# Prepare padded signal using previous buffer
		padded = torch.cat([self.prev, x])
		start = self.prev.shape[0] - n - 1
		if start < 0:
			# pad more on the left
			padded = torch.cat([torch.zeros(-start, device=self.device), padded])
			start = 0
		idx_a = start + torch.arange(0, L, device=self.device)
		idx_b = idx_a + 1
		a = padded[idx_a.long()]
		b = padded[idx_b.long()]
		y = (1.0 - frac) * a + frac * b
		# update prev buffer for next chunk
		tail_len = min(self.max_delay + 2, padded.shape[0])
		self.prev = padded[-tail_len:].clone()
		return y

	def process(self, mono_np: np.ndarray, azimuth: float = 0.0) -> np.ndarray:
		# mono_np: 1D numpy float32
		x = torch.from_numpy(mono_np.astype(np.float32)).to(self.device)
		# map azimuth (-1..1) to max_delay range
		maxd = float(self.max_delay)
		itd = azimuth * maxd  # signed float samples
		# positive => right delayed
		if itd >= 0:
			left = x
			right = self.fractional_delay(x, itd)
		else:
			right = x
			left = self.fractional_delay(x, -itd)

		# IID: simple level difference based on azimuth
		left_gain = 1.0 - 0.25 * max(0.0, azimuth)
		right_gain = 1.0 + 0.25 * min(0.0, azimuth)
		left = left * left_gain
		right = right * right_gain

		stereo = torch.stack([left, right], dim=1).cpu().numpy()
		return stereo

# Fallback numpy processor (existing simple pan)
class NumpyBinauralProcessor:
	def __init__(self, max_delay_ms=2.0, rate=44100):
		self.rate = rate
		self.max_delay = int(rate * max_delay_ms / 1000)

	def process(self, mono_np: np.ndarray, azimuth: float = 0.0) -> np.ndarray:
		# mono -> stereo using integer delay + IID
		L = mono_np.shape[0]
		delay = int(azimuth * self.max_delay)
		if delay > 0:
			left = mono_np
			right = np.concatenate((np.zeros(delay, dtype=np.float32), mono_np[:-delay]))
		elif delay < 0:
			d = -delay
			left = np.concatenate((np.zeros(d, dtype=np.float32), mono_np[:-d]))
			right = mono_np
		else:
			left = mono_np
			right = mono_np
		left_gain = 1.0 - 0.25 * max(0.0, azimuth)
		right_gain = 1.0 + 0.25 * min(0.0, azimuth)
		stereo = np.stack([left * left_gain, right * right_gain], axis=1)
		return stereo


class HRTFProcessor:
	"""Simple streaming HRTF processor.
	- Can load HRIRs from a dict or .npz file (keys: angles or indices)
	- If none provided, generates synthetic short HRIRs using ITD+IID approximations
	- Uses overlap-add for streaming convolution
	"""
	def __init__(self, rate=44100, hrir_dict=None, max_delay_ms=2.0):
		self.rate = rate
		self.max_delay = int(rate * max_delay_ms / 1000)
		self.hrir_dict = hrir_dict or {}
		self.hrirs = None
		if self.hrir_dict:
			# expect {angle: (hL, hR)}
			self.hrirs = sorted([(float(a), np.asarray(v[0], dtype=np.float32), np.asarray(v[1], dtype=np.float32)) for a, v in self.hrir_dict.items()], key=lambda x: x[0])
		# overlap buffers for left/right
		self.overlap_l = np.zeros(0, dtype=np.float32)
		self.overlap_r = np.zeros(0, dtype=np.float32)

	def load_npz(self, path):
		data = np.load(path, allow_pickle=True)
		# Expect keys like 'angle_0', or a dict saved; try a couple formats
		hrir_dict = {}
		if 'hrir' in data:
			# user saved structured array
			hrir = data['hrir']
			# not handling this case generically
		else:
			for k in data.files:
				# try parse 'angle_XX'
				if k.startswith('angle_'):
					ang = float(k.split('_', 1)[1])
					v = data[k]
					if v.shape[1] >= 2:
						hrir_dict[ang] = (v[:,0], v[:,1])
		if hrir_dict:
			self.__init__(rate=self.rate, hrir_dict=hrir_dict, max_delay_ms=self.max_delay*1000/self.rate)

	def _synth_hrir(self, azimuth: float, length=128):
		# create synthetic HRIRs: delta plus small smoothing + delay
		t = np.arange(length)
		# simple exponential decay as simple lowpass behavior
		env = np.exp(-t / (length/8.0)).astype(np.float32)
		# ITD: map azimuth to sample delay
		delay = int(np.round(azimuth * self.max_delay))
		hL = np.zeros(length, dtype=np.float32)
		hR = np.zeros(length, dtype=np.float32)
		if delay >= 0:
			hL[0] = 1.0
			idx = min(length-1, delay)
			hR[idx] = 1.0
		else:
			d = -delay
			idx = min(length-1, d)
			hL[idx] = 1.0
			hR[0] = 1.0
		# apply head-shadow envelope (simple)
		hL *= env * (1.0 - 0.2 * max(0.0, azimuth))
		hR *= env * (1.0 + 0.2 * min(0.0, azimuth))
		return hL, hR

	def _select_hrir(self, azimuth: float):
		if self.hrirs is None:
			return self._synth_hrir(azimuth)
		# find nearest angle
		angles = [a for a,_,_ in self.hrirs]
		idx = int(np.argmin([abs(azimuth - a) for a in angles]))
		return self.hrirs[idx][1], self.hrirs[idx][2]

	def process(self, mono_np: np.ndarray, azimuth: float = 0.0) -> np.ndarray:
		# perform streaming convolution of mono_np with HRIRs
		mono = mono_np.astype(np.float32)
		hL, hR = self._select_hrir(azimuth)
		# time-domain conv
		out_l = np.convolve(mono, hL)
		out_r = np.convolve(mono, hR)
		# add previous overlap
		if self.overlap_l.size > 0:
			out_l[:self.overlap_l.size] += self.overlap_l
		if self.overlap_r.size > 0:
			out_r[:self.overlap_r.size] += self.overlap_r
		# split current chunk sized output; keep tail as overlap
		chunk = mono.shape[0]
		out_len = out_l.shape[0]
		if out_len <= chunk:
			# pad
			out_l = np.pad(out_l, (0, chunk - out_len), mode='constant')
			out_r = np.pad(out_r, (0, chunk - out_len), mode='constant')
			self.overlap_l = np.zeros(0, dtype=np.float32)
			self.overlap_r = np.zeros(0, dtype=np.float32)
			stereo = np.stack([out_l, out_r], axis=1)
			return stereo
		# take first chunk as output and store remainder
		out_chunk_l = out_l[:chunk].astype(np.float32)
		out_chunk_r = out_r[:chunk].astype(np.float32)
		self.overlap_l = out_l[chunk:].astype(np.float32)
		self.overlap_r = out_r[chunk:].astype(np.float32)
		stereo = np.stack([out_chunk_l, out_chunk_r], axis=1)
		return stereo

# 3Dパンニング関数（左右・前後・上下の簡易処理）
def pan_audio(data, azimuth=0.0, elevation=0.0):
	# data: np.ndarray, shape=(n, 2) stereo
	# azimuth: -1.0(左)〜1.0(右), elevation: -1.0(下)〜1.0(上)
	left_gain = 1.0 - max(0, azimuth)
	right_gain = 1.0 + min(0, azimuth)
	# elevationはここでは未使用（HRTF等で拡張可能）
	data[:, 0] *= left_gain
	data[:, 1] *= right_gain
	return data

CHUNK = 1024
FORMAT = pyaudio.paFloat32
RATE = 44100

def open_streams():
	p = pyaudio.PyAudio()
	try:
		in_info = p.get_default_input_device_info()
		out_info = p.get_default_output_device_info()
	except Exception as e:
		print("入力または出力デバイスが見つかりません:", e)
		p.terminate()
		sys.exit(1)

	in_ch = int(in_info.get('maxInputChannels', 0))
	out_ch = int(out_info.get('maxOutputChannels', 0))

	print(f"Default input device: {in_info.get('name')} (maxInputChannels={in_ch})")
	print(f"Default output device: {out_info.get('name')} (maxOutputChannels={out_ch})")

	if in_ch == 0:
		print("入力デバイスがモノラル入力をサポートしていません。終了します。")
		p.terminate()
		sys.exit(1)
	if out_ch == 0:
		print("出力デバイスがありません。終了します。")
		p.terminate()
		sys.exit(1)

	# Open separate input and output streams to avoid channel mismatch errors
	input_stream = p.open(format=FORMAT,
						  channels=in_ch,
						  rate=RATE,
						  input=True,
						  frames_per_buffer=CHUNK)

	output_stream = p.open(format=FORMAT,
						   channels=out_ch,
						   rate=RATE,
						   output=True,
						   frames_per_buffer=CHUNK)

	return p, input_stream, output_stream, in_ch, out_ch


def run():
	global audio3d_state  # 追加
	p, input_stream, output_stream, in_ch, out_ch = open_streams()
	if TORCH_AVAILABLE:
		try:
			binaural = AdvancedBinauralProcessor(rate=RATE)
			use_torch = True
		except Exception as e:
			print('PyTorch available but failed to initialize AdvancedBinauralProcessor:', e)
			binaural = NumpyBinauralProcessor(rate=RATE)
			use_torch = False
	else:
		binaural = NumpyBinauralProcessor(rate=RATE)
		use_torch = False

	print('Binaural processor:', 'PyTorch' if use_torch else 'NumPy fallback')

	print("リアルタイム3D立体音響システム 起動中... Ctrl+Cで終了")

	# Interactive azimuth control (thread-safe)
	import threading
	azimuth_lock = threading.Lock()
	target_azimuth = {'val': 0.0}
	use_hrtf = {'val': False}

	def control_thread():
		print("操作: '<' で左、'>' で右、'q'で終了 (Enter不要)")
		try:
			while True:
				ch = sys.stdin.read(1)
				if not ch:
					break
				with azimuth_lock:
					if ch == '<':
						target_azimuth['val'] = max(-1.0, target_azimuth['val'] - 0.1)
					elif ch == '>':
						target_azimuth['val'] = min(1.0, target_azimuth['val'] + 0.1)
					elif ch.lower() == 'q':
						raise KeyboardInterrupt()
					elif ch.lower() == 'h':
						use_hrtf['val'] = not use_hrtf['val']
						print('HRTF', 'ON' if use_hrtf['val'] else 'OFF')
				print(f"target azimuth={target_azimuth['val']:+.2f}")
		except KeyboardInterrupt:
			return

	ctrl = threading.Thread(target=control_thread, daemon=True)
	try:
		# attempt to set stdin to raw mode on POSIX; on Windows PowerShell this will still work with read(1)
		import msvcrt
		have_msvcrt = True
	except Exception:
		have_msvcrt = False

	if have_msvcrt:
		# On Windows, spawn a small helper that reads keys without blocking the audio loop
		def win_ctrl():
			try:
				while True:
					if msvcrt.kbhit():
						ch = msvcrt.getwch()
						with azimuth_lock:
							if ch == '<':
								target_azimuth['val'] = max(-1.0, target_azimuth['val'] - 0.1)
							elif ch == '>':
								target_azimuth['val'] = min(1.0, target_azimuth['val'] + 0.1)
							elif ch.lower() == 'q':
								raise KeyboardInterrupt()
							elif ch.lower() == 'h':
								use_hrtf['val'] = not use_hrtf['val']
								print('HRTF', 'ON' if use_hrtf['val'] else 'OFF')
						print(f"target azimuth={target_azimuth['val']:+.2f}")
					else:
						time.sleep(0.01)
			except KeyboardInterrupt:
				return

		ctrl = threading.Thread(target=win_ctrl, daemon=True)

	ctrl.start()

	# smoothing state
	current_az = 0.0
	smoothing_alpha = 0.2
	try:
		while True:
			in_data = input_stream.read(CHUNK, exception_on_overflow=False)
			audio = np.frombuffer(in_data, dtype=np.float32)
			audio = audio.reshape(-1, in_ch)

			# Convert input -> output channel count
			if in_ch == 1 and out_ch == 2:
				audio = np.repeat(audio, 2, axis=1)
			elif in_ch == 2 and out_ch == 1:
				audio = audio.mean(axis=1, keepdims=True)
			elif in_ch != out_ch:
				if in_ch > out_ch:
					audio = audio[:, :out_ch]
				else:
					pad_width = out_ch - in_ch
					audio = np.pad(audio, ((0, 0), (0, pad_width)), mode='constant')

			# 出力立体音響OFFなら無音（ゼロ配列）を出力
			if not audio3d_state['output']:
				audio = np.zeros_like(audio)
			else:
				# If microphone monitoring is disabled, silence the mic before processing
				if not audio3d_state.get('monitor_mic', False):
					# replace microphone input with silence but keep shape/channel count
					audio = np.zeros_like(audio)
				else:
					# If output is stereo, apply processing
					if out_ch == 2:
						with azimuth_lock:
							tgt = target_azimuth['val']
						current_az = (1.0 - smoothing_alpha) * current_az + smoothing_alpha * tgt
						mono = audio.mean(axis=1)
						if use_hrtf['val']:
							if not hasattr(run, '_hrtf'):
								run._hrtf = HRTFProcessor(rate=RATE)
							stereo = run._hrtf.process(mono, azimuth=current_az)
						else:
							stereo = binaural.process(mono, azimuth=current_az)
						audio = stereo

			output_stream.write(audio.astype(np.float32).tobytes())
	except KeyboardInterrupt:
		print('\n停止中...')
	finally:
		input_stream.stop_stream()
		input_stream.close()
		output_stream.stop_stream()
		output_stream.close()
		p.terminate()


import threading
import customtkinter as ctk

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# グローバルで状態を持つ
audio3d_state = {
	'input': False,
	'output': False,
	# whether microphone input should be audible in the output (monitoring)
	'monitor_mic': False
}

class Audio3DControl(ctk.CTk):
	def __init__(self):
		super().__init__()
		self.title("3D Audio Control")
		self.geometry("300x180")

		self.input_var = ctk.BooleanVar(value=False)
		self.output_var = ctk.BooleanVar(value=False)

		self.label = ctk.CTkLabel(self, text="立体音響 オン/オフ")
		self.label.pack(pady=10)

		self.input_checkbox = ctk.CTkCheckBox(
			self, text="入力に立体音響を適用", variable=self.input_var)
		self.input_checkbox.pack(pady=10)

		self.output_checkbox = ctk.CTkCheckBox(
			self, text="出力に立体音響を適用", variable=self.output_var)
		self.output_checkbox.pack(pady=10)

		self.monitor_var = ctk.BooleanVar(value=False)
		self.monitor_checkbox = ctk.CTkCheckBox(
			self, text="マイクを出力で聞く（モニター）", variable=self.monitor_var)
		self.monitor_checkbox.pack(pady=10)

		self.status_label = ctk.CTkLabel(self, text="")
		self.status_label.pack(pady=10)

		self.input_var.trace_add("write", self.update_status)
		self.output_var.trace_add("write", self.update_status)
		# monitor checkbox state update
		self.monitor_var.trace_add("write", self.update_status)

	def update_status(self, *args):
		# グローバル状態に反映
		audio3d_state['input'] = self.input_var.get()
		audio3d_state['output'] = self.output_var.get()
		audio3d_state['monitor_mic'] = bool(self.monitor_var.get())
		status = f"入力: {'ON' if self.input_var.get() else 'OFF'} / 出力: {'ON' if self.output_var.get() else 'OFF'} / マイク監視: {'ON' if audio3d_state['monitor_mic'] else 'OFF'}"
		self.status_label.configure(text=status)


if __name__ == "__main__":
	# Start audio processing in a background thread and run GUI in the main thread
	audio_thread = threading.Thread(target=run, daemon=True)
	audio_thread.start()
	Audio3DControl().mainloop()
