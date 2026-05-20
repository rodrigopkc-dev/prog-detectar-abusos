# Detecção de Abusos em Vídeos com IA

Este repositório contém um projeto de análise de conteúdo multimídia que combina:

- Reconhecimento de fala e análise de sentimento com serviços AWS (Transcribe + Comprehend).
- Detecção de padrões de risco em vídeo usando YOLO local para identificar expressões faciais e emoções.
- Interface de visualização com Streamlit para consolidar os resultados.
- Na pasta "/video" possui dois videos de testes, um video que demonstra felicidade e outro video que possuis sinais de desconforto.
- Para o programa acessar os serviços da AWS será necessário ter: AWS_ACCESS_KEY_ID e AWS_SECRET_ACCESS_KEY.
- Link Yolo 8 treinado: "https://drive.google.com/file/d/1Ia8txfBiCn4Z8YC3hpCdt6wQsaiy3YSv/view?usp=sharing"



## Visão Geral

O projeto processa vídeos em duas frentes:

1. `aws_codigo_completo.py`
   - Extrai o áudio do vídeo.
   - Faz upload para um bucket AWS S3.
   - Executa transcrição com AWS Transcribe.
   - Analisa sentimento do texto transcrito com AWS Comprehend.

2. `yolo_local.py`
   - Executa detecção local de expressões faciais usando um modelo YOLO customizado.
   - Identifica emoções associadas a risco, como `Fear`, `Sad` e `Angry`.
   - Gera log de alertas e calcula índices de risco.

3. `principal.py`
   - Interface Streamlit que recebe vídeo do usuário.
   - Dispara o processamento AWS + YOLO.
   - Exibe resultados consolidados em painel web.

## Estrutura do Repositório

- `principal.py` - Aplicação Streamlit principal.
- `aws_codigo_completo.py` - Funções AWS, transcrição e análise de sentimento.
- `yolo_local.py` - Funções de análise de vídeo e detecção de risco local.
- `audio/` - Pasta de saída de áudio gerado a partir do vídeo.
- `video/` - Pasta de vídeos usados/testados.
- `log_risco_*.txt` - Logs de risco gerados pelo pipeline YOLO.

## Requisitos

- Python 3.10+ recomendado
- Conta AWS com permissões para:
  - S3
  - Transcribe
  - Comprehend
- `ffmpeg` instalado no sistema para o MoviePy funcionar corretamente.

## Dependências Python

Instale as dependências necessárias:

```bash
pip install boto3 python-dotenv moviepy streamlit ultralytics opencv-python
```

> Dependendo da sua instalação, `opencv-python` pode precisar de pacotes adicionais do sistema.

## Configuração

Crie um arquivo `.env` na raiz do repositório com as credenciais AWS e o bucket S3:

```env
CHAVE=<AWS_ACCESS_KEY_ID>
SECRET=<AWS_SECRET_ACCESS_KEY>
BUCKET_NAME=<NOME_DO_BUCKET>
```

A aplicação usa as variáveis:

- `CHAVE` para AWS Access Key ID.
- `SECRET` para AWS Secret Access Key.
- `BUCKET_NAME` para o nome do bucket S3.

## Executando a Aplicação

Execute a interface Streamlit:

```bash
streamlit run principal.py
```

Depois disso, abra o endereço exibido no terminal e carregue um arquivo MP4.

## Como Funciona

### Pipeline AWS (`aws_codigo_completo.py`)

- `obter_clientes_aws()` cria clientes `boto3` para S3, Transcribe e Comprehend.
- `extrair_audio()` extrai áudio do vídeo usando `moviepy`.
- `upload_para_s3()` envia o áudio para S3.
- `ejecutar_transcricao()` inicia um job do AWS Transcribe e aguarda a conclusão.
- `analisar_sentimento_transcricao()` lê o JSON de saída no S3 e extrai o sentimento.

### Pipeline YOLO Local (`yolo_local.py`)

- `analisar_video_yolo()` abre o vídeo com `cv2` e executa detecção frame a frame.
- O modelo é carregado via `ultralytics.YOLO` com um caminho de peso customizado.
- O código conta eventos de emoção e gera um `log_risco_<nome>.txt`.
- O veredito considera a presença e persistência dos alertas.

## Personalização

### Ajustar modelo YOLO

Para o programa executar o YOLOv8 treinado com sucesso:

- Faça o download do modelo YOLO treinado do link: `https://drive.google.com/file/d/1Ia8txfBiCn4Z8YC3hpCdt6wQsaiy3YSv/view?usp=sharing`
- Extraia os arquivos do modelo em uma pasta
- No arquivo `yolo_local.py`, atualize o caminho do YOLOv8 treinado para onde foi feita a extração
- Default: `model_path=r'D:\notebook2\python\yolov8_facial_FAST_20260508_2217_medio\weights\best.pt'`

O caminho do modelo está atualmente hardcoded em `yolo_local.py`:

```python
model_path=r'D:\notebook2\python\yolov8_facial_FAST_20260508_2217_medio\weights\best.pt'
```

Substitua esse caminho pelo local do seu arquivo `.pt` ou modifique a função para receber esse valor dinamicamente.

### Configurar limites de risco

Os parâmetros de risco estão definidos em:

- `conf=0.42` no `model.predict(...)`
- `score_risco >= 90`
- `taxa_presenca_alerta > 18`
- `score_risco > 25`
- `taxa_presenca_alerta > 4`

Ajuste esses valores conforme a sensibilidade desejada.

## Saída

A aplicação gera:

- Relatório visual na interface Streamlit.
- Logs de análise de risco `log_risco_<nome>.txt` no diretório do projeto.
- Áudio extraído em `audio/audio.mp3`.

## Observações Importantes

- O projeto depende de serviços AWS pagos. Use com cautela e monitore o consumo.
- A análise local de vídeo pressupõe que o modelo YOLO foi treinado para emoções específicas.
- Ajuste parâmetros e caminho de modelo antes de rodar em produção.

## Melhorias Futuras

- Tornar o caminho do modelo configurável via `.env` ou parâmetro.
- Salvar resultados consolidados em um JSON ou banco de dados.
- Adicionar tratamento de exceções mais robusto para AWS e vídeo.
- Suporte a múltiplos formatos de vídeo e processamento assíncrono.

## Contato

Caso queira migrar ou evoluir o projeto, este README ajuda a entender os fluxos principais e a configuração inicial.
