Reconhecimento Facial com Cadastro Biométrico (OpenCV + LBPH)

Sistema de **reconhecimento facial em tempo real** com **fluxo completo de cadastro biométrico**, captura automática de imagens via webcam, **pré-processamento**, **treinamento com 1 clique** e ajuste de threshold por interface gráfica.

Este projeto foi desenvolvido em **Python** usando **OpenCV + LBPH** e **Tkinter**, com foco educacional e de portfólio, simulando uma experiência próxima de soluções reais de onboarding biométrico.

---

## ✨ Destaques

- ✅ Interface gráfica simples e funcional
- ✅ **Cadastro de nova pessoa** com captura automática de **20 fotos**
- ✅ Dataset organizado automaticamente em `dataset/`
- ✅ Pré-processamento de face:
  - detecção com Haar Cascade
  - padronização para **200x200**
  - **equalização de histograma**
- ✅ **Treinamento automático** do modelo LBPH
- ✅ Reconhecimento em tempo real
- ✅ **Slider de threshold** para ajustar sensibilidade
- ✅ Aviso de **dataset desbalanceado**
- ✅ Estrutura preparada para uso em `.py` e empacotamento `.exe` (PyInstaller)

---

## 🧠 Como funciona

1. Você cadastra uma pessoa pelo botão **"Adicionar nova pessoa"**
2. O sistema abre a webcam e **captura automaticamente** as imagens do rosto
3. As faces já são salvas pré-processadas no dataset
4. Com **"Treinar e iniciar reconhecimento"**, o sistema:
   - carrega o dataset
   - treina o LBPH
   - inicia o reconhecimento ao vivo

> No LBPH: **menor confidence = melhor match**  
> Recomendação prática: use threshold entre **60 e 70** para reduzir confusões.

---

## 🧰 Tecnologias

- Python
- OpenCV (opencv-contrib)
- Tkinter
- NumPy

---

🖥️ Controles principais

Na interface:
 - Adicionar nova pessoa (capturar 20 fotos)
      - solicita o nome
      - captura e salva automaticamente
 - Treinar e iniciar reconhecimento
      - treina o modelo e abre a janela de reconhecimento
 - Slider de threshold
      - ajusta o nível de confiança do LBPH

No reconhecimento:
      - Pressione q para encerrar a janela da webcam.

---

📌 Observações sobre qualidade do modelo

Para melhorar a assertividade:
 - mantenha quantidades similares de fotos por pessoa
 - cadastre fotos com:
    - variações leves de ângulo
    - expressões naturais
    - iluminação diferente
 - evite rostos muito pequenos ou desfocados
O próprio sistema exibe um aviso quando detecta desbalanceamento significativo.

---

🔒 Uso responsável

Este repositório tem propósito educacional e demonstrativo.
Para aplicações reais:
    - obtenha consentimento explícito
    - implemente controle seguro de armazenamento
    - avalie modelos modernos baseados em embeddings

---

##👤 Autor

Mickael Lopes de Souza
Projeto de segurança e qualidade em Visão Computacional e ML aplicado.
![Reconheicmento_Facial_1](https://github.com/user-attachments/assets/6db8c289-1800-4946-bf79-899a79dcecb1)
![Reconheicmento_Facial_2](https://github.com/user-attachments/assets/46ed8846-f1eb-4934-8948-6c3233d3b3fa)


## 📦 Instalação

```bash
pip install opencv-contrib-python numpy

  ▶️ Execução
python Reconhecimento_Facial.py

  📁 Estrutura do projeto
reconhecimento-facial-cadastro-lbph/
│
├── Reconhecimento_Facial.py
└── dataset/
    ├── nome_01.jpg
    ├── nome_02.jpg
    └── ...


