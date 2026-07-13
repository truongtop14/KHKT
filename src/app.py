# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import threading
import sys
import os
import whisper
import pandas as pd
import numpy as np
import re

# ---------------------------------------------------------
# Lớp hỗ trợ chuyển hướng lệnh 'print' vào giao diện Tkinter
# ---------------------------------------------------------
class RedirectText(object):
    def __init__(self, text_widget):
        self.output = text_widget

    def write(self, string):
        self.output.insert(tk.END, string)
        self.output.see(tk.END)

    def flush(self):
        pass

# ---------------------------------------------------------
# Logic xử lý chính
# ---------------------------------------------------------
def transcribe_audio(model, audio_path, output_csv="data/transcript.csv"):
    print(f"📥 Đang xử lý file audio: {audio_path}")
    print("⏳ Vui lòng đợi, quá trình nhận diện giọng nói có thể mất vài phút...")
    
    result = model.transcribe(audio_path, word_timestamps=True)

    rows = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            rows.append([
                w.get("word", "").strip(),
                w.get("probability", 0),
                w.get("start", 0),
                w.get("end", 0)
            ])

    df = pd.DataFrame(rows, columns=["word", "probability", "start", "end"])
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"✅ Đã lưu file transcript vào {output_csv}")
    return True

def fluency_feedback(csv_path="data/transcript.csv"):
    df = pd.read_csv(csv_path)
    total_duration = df.iloc[-1]['end']
    total_words = len(df)
    speech_rate_wpm = total_words / total_duration * 60

    pauses = []
    for i in range(1, total_words):
        gap = df.iloc[i]['start'] - df.iloc[i-1]['end']
        if gap >= 0.25:
            pauses.append(gap)

    pause_ratio = sum(pauses) / total_duration

    print("\n=== FLUENCY FEEDBACK ===")
    print(f"Speech rate: {speech_rate_wpm:.2f} WPM")
    print(f"Pause ratio: {pause_ratio:.2f}")

    if speech_rate_wpm >= 90 and pause_ratio <= 0.3:
        print("✔ Tốc độ nói tốt và kiểm soát khoảng nghỉ tốt.")
    elif speech_rate_wpm >= 90:
        print("⚠ Nói nhanh nhưng có quá nhiều khoảng nghỉ.")
    elif pause_ratio > 0.3:
        print("⚠ Nói chậm và có nhiều khoảng nghỉ.")
    else:
        print("⚠ Nói chậm nhưng khoảng nghỉ chấp nhận được.")

def tokenize(words):
    tokens = []
    for w in words:
        w = str(w).lower()
        w = re.sub(r"[^a-z']", "", w)
        if w != "":
            tokens.append(w)
    return tokens

def compute_ttr(tokens):
    if len(tokens) == 0: return 0
    return len(set(tokens)) / len(tokens)

def compute_msttr(tokens, segment_size=50):
    if len(tokens) < segment_size:
        return compute_ttr(tokens)
    ttrs = []
    for i in range(0, len(tokens) - segment_size + 1, segment_size):
        segment = tokens[i:i+segment_size]
        ttrs.append(compute_ttr(segment))
    return float(np.mean(ttrs))

def lexical_diversity(csv_path="data/transcript.csv"):
    df = pd.read_csv(csv_path)
    tokens = tokenize(df['word'])
    ttr = compute_ttr(tokens)
    msttr = compute_msttr(tokens, segment_size=50)

    print("\n=== LEXICAL DIVERSITY ===")
    print(f"TTR: {ttr:.4f}")
    print(f"MSTTR: {msttr:.4f}")
    if msttr > 0.7:
        print("✔ Sử dụng từ vựng phong phú.")
    else:
        print("⚠ Từ vựng hạn chế, nên sử dụng nhiều từ đa dạng hơn.")

def load_cefr(oxford_file):
    df = pd.read_csv(oxford_file)
    return dict(zip(df['word'], df['level']))

def lexical_cefr(csv_path="data/transcript.csv", cefr_path="data/oxford_cerf.csv"):
    if not os.path.exists(cefr_path):
        print(f"\n❌ Lỗi: Không tìm thấy file từ vựng {cefr_path}")
        return

    df = pd.read_csv(csv_path)
    cefr = load_cefr(cefr_path)

    levels = []
    for w in df['word']:
        w = re.sub(r"[^a-zA-Z']", "", str(w).lower())
        levels.append(cefr.get(w, "A1"))

    a1 = sum(l == 'A1' for l in levels)
    a2 = sum(l == 'A2' for l in levels)
    b1 = sum(l == 'B1' for l in levels)
    b2 = sum(l == 'B2' for l in levels)
    c1 = sum(l == 'C1' for l in levels)

    print("\n=== CEFR VOCABULARY ===")
    print(f"A1: {a1/len(levels)*100:.1f}%")
    print(f"A2: {a2/len(levels)*100:.1f}%")
    print(f"B1: {b1/len(levels)*100:.1f}%")
    print(f"B2: {b2/len(levels)*100:.1f}%")
    print(f"C1: {c1/len(levels)*100:.1f}%")

    if (b1+b2+c1)/len(levels)*100 > 30:
        print("✔ Sử dụng nhiều từ vựng nâng cao.")
    else:
        print("⚠ Chủ yếu sử dụng từ vựng cơ bản.")

def pronunciation_feedback(csv_path="data/transcript.csv"):
    df = pd.read_csv(csv_path)
    high = (df['probability'] >= 0.85).mean() * 100
    mid = ((df['probability'] < 0.85) & (df['probability'] >= 0.5)).mean() * 100
    low = (df['probability'] < 0.5).mean() * 100

    print("\n=== PRONUNCIATION ===")
    print(f"Độ tự tin cao: {high:.1f}%")
    print(f"Độ tự tin trung bình: {mid:.1f}%")
    print(f"Độ tự tin thấp: {low:.1f}%")

    if high > 60:
        print("✔ Phát âm rõ ràng.")
    else:
        print("⚠ Phát âm cần cải thiện.")

def grammar_feedback(csv_path="data/transcript.csv"):
    print("\n⏳ Đang kiểm tra ngữ pháp với LanguageTool...")
    try:
        import language_tool_python
        # Khởi tạo LanguageTool cho tiếng Anh Mỹ
        tool = language_tool_python.LanguageTool('en-US')
        
        df = pd.read_csv(csv_path)
        # Nối các từ lại thành một đoạn văn hoàn chỉnh
        full_text = " ".join(df['word'].astype(str).tolist())
        
        # Sửa lỗi khoảng trắng thừa trước dấu câu (do join)
        full_text = re.sub(r'\s+([?.!,"])', r'\1', full_text)
        
        matches = tool.check(full_text)
        
        total_words = len(df)
        error_count = len(matches)
        
        print("\n=== GRAMMAR FEEDBACK ===")
        print(f"Tổng số lỗi phát hiện: {error_count}")
        
        if error_count == 0:
            print("✔ Ngữ pháp rất tốt, không tìm thấy lỗi.")
        else:
            print("⚠ Dưới đây là một số lỗi cơ bản cần lưu ý:")
            # Chỉ in ra tối đa 5 lỗi để giao diện không bị rối
            for i, match in enumerate(matches[:5]):
                print(f"  [{i+1}] {match.message}")
                print(f"      Từ sai: '{match.context[match.offset:match.offset+match.errorLength]}'")
                if match.replacements:
                    print(f"      Gợi ý sửa: {match.replacements[:3]}")
                print("-" * 30)
            
            if error_count > 5:
                print(f"  ... và {error_count - 5} lỗi khác.")
        
        tool.close()

    except ImportError:
        print("❌ Lỗi: Chưa cài đặt thư viện. Vui lòng chạy lệnh: pip install language-tool-python")
    except Exception as e:
        print(f"❌ Lỗi khi kiểm tra ngữ pháp: {str(e)}")

# ---------------------------------------------------------
# Xây dựng giao diện Tkinter
# ---------------------------------------------------------
class AudioFeedbackApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Speaking Feedback Pro")
        self.root.geometry("750x650")
        
        os.makedirs("data", exist_ok=True)
        
        self.audio_path = None
        self.model = None

        self.setup_ui()

        redir = RedirectText(self.text_output)
        sys.stdout = redir
        
        print("Ứng dụng đã sẵn sàng. Vui lòng chọn file âm thanh để bắt đầu.\n")

    def setup_ui(self):
        control_frame = tk.Frame(self.root, pady=10)
        control_frame.pack(fill=tk.X)

        self.btn_select = tk.Button(control_frame, text="1. Chọn File Audio", command=self.select_file, width=20, font=("Arial", 10, "bold"))
        self.btn_select.pack(side=tk.LEFT, padx=10)

        self.lbl_file = tk.Label(control_frame, text="Chưa chọn file nào", fg="gray")
        self.lbl_file.pack(side=tk.LEFT, padx=10)

        self.btn_run = tk.Button(control_frame, text="2. Phân Tích", command=self.start_analysis, width=15, state=tk.DISABLED, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.btn_run.pack(side=tk.RIGHT, padx=10)

        self.text_output = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, font=("Consolas", 10))
        self.text_output.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    def select_file(self):
        filetypes = (
            ("Audio files", "*.mp3 *.wav *.m4a *.flac"),
            ("All files", "*.*")
        )
        filename = filedialog.askopenfilename(title="Chọn file âm thanh", filetypes=filetypes)
        if filename:
            self.audio_path = filename
            self.lbl_file.config(text=os.path.basename(filename), fg="black")
            self.btn_run.config(state=tk.NORMAL)

    def start_analysis(self):
        if not self.audio_path:
            return
            
        self.btn_select.config(state=tk.DISABLED)
        self.btn_run.config(state=tk.DISABLED)
        self.text_output.delete(1.0, tk.END)
        
        threading.Thread(target=self.run_logic, daemon=True).start()

    def run_logic(self):
        try:
            if self.model is None:
                print("⏳ Lần đầu chạy: Đang nạp mô hình Whisper (small.en)...")
                self.model = whisper.load_model("small.en")
                print("✅ Tải mô hình thành công!\n")

            csv_path = "data/transcript.csv"
            cefr_path = "data/oxford_cerf.csv"
            
            success = transcribe_audio(self.model, self.audio_path, csv_path)
            
            if success:
                print("\n" + "="*40)
                print("=== TỔNG HỢP OVERALL FEEDBACK ===")
                print("="*40)
                
                fluency_feedback(csv_path)
                lexical_diversity(csv_path)
                lexical_cefr(csv_path, cefr_path)
                pronunciation_feedback(csv_path)
                
                # Gọi thêm hàm phân tích ngữ pháp
                grammar_feedback(csv_path)
                
                print("="*40 + "\n✅ HOÀN TẤT TẤT CẢ CÁC BƯỚC!")

        except Exception as e:
            print(f"\n❌ Đã xảy ra lỗi: {str(e)}")
        
        finally:
            self.root.after(0, lambda: self.btn_select.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL))

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioFeedbackApp(root)
    root.mainloop()