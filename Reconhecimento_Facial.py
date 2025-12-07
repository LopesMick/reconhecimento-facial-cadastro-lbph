import os
import sys
import threading
import time
import re
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from collections import Counter

# ===== BASE_DIR funciona tanto em .py quanto em .exe (PyInstaller) =====
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Garantir que a pasta dataset exista
os.makedirs(DATASET_DIR, exist_ok=True)

# ===== DETECTOR =====
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ===== CONFIGS =====
# No LBPH: menor confidence = melhor match
CONFIDENCE_THRESHOLD = 65  # padrão mais seguro
DEBUG_PREDICTIONS = False  # deixe True se quiser ver logs no CMD

CAPTURE_TARGET = 20  # número de fotos a capturar por pessoa


# ==========================
# Utils
# ==========================
def sanitize_name(name: str) -> str:
    """
    Sanitiza o nome para virar parte segura de filename.
    - remove espaços extras
    - troca espaços por vazio
    - mantém letras/números/underscore
    """
    name = name.strip()
    name = name.replace(" ", "")
    # remover caracteres não seguros
    name = re.sub(r"[^A-Za-z0-9_]", "", name)
    return name


def equalize_face(gray_face):
    """Equalização simples de histograma."""
    return cv2.equalizeHist(gray_face)


def preprocess_face_from_gray(gray, x=None, y=None, w=None, h=None):
    """
    Recebe uma imagem gray e opcionalmente uma ROI.
    Devolve face 200x200 equalizada.
    """
    if x is not None:
        face_roi = gray[y:y + h, x:x + w]
    else:
        face_roi = gray

    face_roi = cv2.resize(face_roi, (200, 200))
    face_roi = equalize_face(face_roi)
    return face_roi


# ==========================
# ETAPA 1: Carregar dataset
# ==========================
def carregar_dataset():
    faces = []
    labels = []
    nomes = []

    if not os.path.isdir(DATASET_DIR):
        print(f"[ERRO] Pasta dataset não encontrada: {DATASET_DIR}")
        return faces, labels, nomes

    arquivos = sorted(os.listdir(DATASET_DIR))

    for arquivo in arquivos:
        if not (arquivo.lower().endswith(".jpg")
                or arquivo.lower().endswith(".png")
                or arquivo.lower().endswith(".jpeg")):
            continue

        caminho_imagem = os.path.join(DATASET_DIR, arquivo)

        nome = arquivo.split("_")[0]
        if nome not in nomes:
            nomes.append(nome)
        label_id = nomes.index(nome)

        img = cv2.imread(caminho_imagem)
        if img is None:
            print(f"[AVISO] Não foi possível ler a imagem: {caminho_imagem}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Tenta detectar face
        faces_detectadas = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces_detectadas) == 0:
            # ===== MELHORIA: se não detectar, assume que a imagem já é um rosto recortado
            face_roi = preprocess_face_from_gray(gray)
            faces.append(face_roi)
            labels.append(label_id)
            print(f"[OK] Treino (recorte direto): {arquivo} -> nome: {nome}, label: {label_id}")
            continue

        (x, y, w, h) = faces_detectadas[0]
        face_roi = preprocess_face_from_gray(gray, x, y, w, h)

        faces.append(face_roi)
        labels.append(label_id)

        print(f"[OK] Treino: {arquivo} -> nome: {nome}, label: {label_id}")

    # Diagnóstico de balanceamento
    if labels:
        contagem = Counter([nomes[l] for l in labels])
        print("\n[INFO] Quantidade de imagens usadas por pessoa:")
        for n, qtd in contagem.items():
            print(f" - {n}: {qtd}")

        if len(contagem) > 1:
            max_qtd = max(contagem.values())
            min_qtd = min(contagem.values())
            if max_qtd >= (min_qtd * 2):
                print("[AVISO] Dataset desbalanceado detectado.")
                print("        Tente manter quantidades parecidas por pessoa.\n")

    return faces, labels, nomes


# ==========================
# ETAPA 2: Treinar LBPH
# ==========================
def treinar_modelo(faces, labels):
    labels = np.array(labels)

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )

    print("\n[INFO] Treinando modelo LBPH (melhorado)...")
    recognizer.train(faces, labels)
    print("[INFO] Treino concluído.\n")

    return recognizer


# ==========================
# ETAPA 3: Reconhecer webcam
# ==========================
def reconhecer_webcam(recognizer, nomes):
    global CONFIDENCE_THRESHOLD

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Erro", "Não foi possível acessar a webcam.")
        return

    print("[INFO] Webcam iniciada. Pressione 'q' na janela de vídeo para encerrar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces_detectadas = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        for (x, y, w, h) in faces_detectadas:
            face_roi = preprocess_face_from_gray(gray, x, y, w, h)

            label_id, confidence = recognizer.predict(face_roi)

            if DEBUG_PREDICTIONS:
                nome_pred = nomes[label_id] if 0 <= label_id < len(nomes) else "?"
                print(f"[DEBUG] Pred={nome_pred} | confidence={confidence:.2f} | threshold={CONFIDENCE_THRESHOLD}")

            if confidence < CONFIDENCE_THRESHOLD and 0 <= label_id < len(nomes):
                nome = nomes[label_id]
                texto = f"{nome} ({confidence:.1f})"
            else:
                texto = f"Desconhecido ({confidence:.1f})"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y - 25), (x + w, y), (0, 255, 0), cv2.FILLED)
            cv2.putText(
                frame,
                texto,
                (x + 5, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow("Reconhecimento Facial - OpenCV + LBPH (Cadastro Profissional)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam encerrada.")


# ==========================
# CAPTURA AUTOMÁTICA DE NOVA PESSOA
# ==========================
def capturar_nova_pessoa(nome_pessoa, target=CAPTURE_TARGET):
    """
    Captura automática de N fotos com a webcam.
    Salva no dataset como imagens do rosto já recortadas (200x200 grays).
    """
    nome_pessoa = sanitize_name(nome_pessoa)
    if not nome_pessoa:
        messagebox.showwarning("Atenção", "Nome inválido. Use apenas letras e números.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Erro", "Não foi possível acessar a webcam.")
        return

    # Descobrir próximo índice disponível para não sobrescrever
    existentes = [f for f in os.listdir(DATASET_DIR) if f.startswith(nome_pessoa + "_")]
    idx_start = 1
    if existentes:
        # tenta extrair maior número existente
        nums = []
        for f in existentes:
            m = re.search(rf"^{re.escape(nome_pessoa)}_(\d+)", f)
            if m:
                nums.append(int(m.group(1)))
        if nums:
            idx_start = max(nums) + 1

    capturadas = 0
    last_save_time = 0

    print(f"[INFO] Iniciando captura de {target} fotos para: {nome_pessoa}")
    print("[INFO] Olhe para a câmera com boa luz. Variações leves de ângulo ajudam.")

    while capturadas < target:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces_detectadas = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        # Interface de feedback na janela
        display = frame.copy()
        cv2.putText(
            display,
            f"Cadastrando: {nome_pessoa} | {capturadas}/{target}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        if len(faces_detectadas) > 0:
            # pega maior face detectada
            faces_sorted = sorted(faces_detectadas, key=lambda b: b[2] * b[3], reverse=True)
            x, y, w, h = faces_sorted[0]

            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Controle simples para não salvar frames idênticos muito rápido
            now = time.time()
            if now - last_save_time > 0.25:
                face_roi = preprocess_face_from_gray(gray, x, y, w, h)

                filename = f"{nome_pessoa}_{idx_start + capturadas:02d}.jpg"
                path_out = os.path.join(DATASET_DIR, filename)

                cv2.imwrite(path_out, face_roi)
                capturadas += 1
                last_save_time = now

        cv2.imshow("Cadastro Biométrico - Captura Automática", display)

        # Pressione 'q' para abortar cadastro
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if capturadas >= target:
        messagebox.showinfo("Cadastro concluído",
                            f"Foram capturadas {capturadas} imagens para '{nome_pessoa}'.")
    else:
        messagebox.showwarning("Cadastro interrompido",
                               f"Cadastro interrompido. Capturadas {capturadas}/{target} imagens.")


# ==========================
# FLUXO COMPLETO
# ==========================
def fluxo_reconhecimento():
    print("[INFO] Carregando dataset...")
    faces, labels, nomes = carregar_dataset()

    if not faces:
        messagebox.showerror("Erro", "Nenhuma face foi carregada.\nVerifique a pasta 'dataset'.")
        return

    recognizer = treinar_modelo(faces, labels)
    reconhecer_webcam(recognizer, nomes)


# ==========================
# GUI
# ==========================
def atualizar_threshold(valor):
    global CONFIDENCE_THRESHOLD
    try:
        CONFIDENCE_THRESHOLD = float(valor)
    except ValueError:
        pass


def iniciar_reconhecimento_thread():
    thread = threading.Thread(target=fluxo_reconhecimento, daemon=True)
    thread.start()


def cadastrar_pessoa_thread():
    nome = simpledialog.askstring("Adicionar nova pessoa",
                                  "Digite o nome da pessoa (sem acentos):")
    if not nome:
        return

    nome = sanitize_name(nome)
    if not nome:
        messagebox.showwarning("Atenção", "Nome inválido. Use letras, números ou underscore.")
        return

    # Captura em thread para não travar GUI
    def _worker():
        capturar_nova_pessoa(nome, CAPTURE_TARGET)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()


def criar_interface():
    root = tk.Tk()
    root.title("Trabalho 4 - Reconhecimento Facial (Cadastro Profissional)")
    root.resizable(False, False)

    frame = ttk.Frame(root, padding=15)
    frame.grid(row=0, column=0, sticky="nsew")

    lbl_titulo = ttk.Label(
        frame,
        text="Reconhecimento Facial - OpenCV + LBPH",
        font=("Segoe UI", 12, "bold")
    )
    lbl_titulo.grid(row=0, column=0, columnspan=2, pady=(0, 10))

    lbl_dataset = ttk.Label(
        frame,
        text=f"Dataset: {DATASET_DIR}",
        wraplength=440,
        justify="left"
    )
    lbl_dataset.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 10))

    # Threshold
    lbl_thr = ttk.Label(frame, text="Nível de confiança (threshold):")
    lbl_thr.grid(row=2, column=0, sticky="w")

    scale_thr = ttk.Scale(
        frame,
        from_=40,
        to=90,
        orient="horizontal",
        command=atualizar_threshold
    )
    scale_thr.set(CONFIDENCE_THRESHOLD)
    scale_thr.grid(row=2, column=1, sticky="ew", padx=(10, 0))

    lbl_thr_info = ttk.Label(
        frame,
        text="LBPH: menor confidence = melhor match.\n"
             "Recomendado: 60 a 70 para evitar confusões.",
        justify="left"
    )
    lbl_thr_info.grid(row=3, column=0, columnspan=2, sticky="w", pady=(5, 12))

    # Botões
    btn_cadastrar = ttk.Button(
        frame,
        text=f"Adicionar nova pessoa (capturar {CAPTURE_TARGET} fotos)",
        command=cadastrar_pessoa_thread
    )
    btn_cadastrar.grid(row=4, column=0, columnspan=2, pady=(0, 8), sticky="ew")

    btn_iniciar = ttk.Button(
        frame,
        text="Treinar e iniciar reconhecimento",
        command=iniciar_reconhecimento_thread
    )
    btn_iniciar.grid(row=5, column=0, columnspan=2, pady=(0, 8), sticky="ew")

    btn_sair = ttk.Button(frame, text="Sair", command=root.destroy)
    btn_sair.grid(row=6, column=0, columnspan=2, sticky="ew")

    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(1, weight=1)

    root.mainloop()


if __name__ == "__main__":
    criar_interface()
