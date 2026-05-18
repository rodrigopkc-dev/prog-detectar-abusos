import os
import time
import uuid
import json
import boto3
from dotenv import load_dotenv
from moviepy import VideoFileClip
from botocore.exceptions import NoCredentialsError, ClientError

# Carrega as variáveis de ambiente
load_dotenv()

# Configurações Globais
AWS_ACCESS_KEY_ID = os.getenv('CHAVE')
AWS_SECRET_ACCESS_KEY = os.getenv('SECRET')
AWS_REGION = "us-east-2"
BUCKET_NAME = os.getenv('BUCKET_NAME')

LOCAL_AUDIO_PATH = "audio/audio.mp3"
S3_AUDIO_KEY = "uploads/audio.mp3"
OUTPUT_KEY_PREFIX = "transcriptions/"

def obter_clientes_aws():
    config = {
        "aws_access_key_id": AWS_ACCESS_KEY_ID,
        "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
        "region_name": AWS_REGION
    }
    s3 = boto3.client("s3", **config)
    transcribe = boto3.client("transcribe", **config)
    comprehend = boto3.client("comprehend", **config)
    return s3, transcribe, comprehend

def extrair_audio(video_path, audio_output_path):
    print(f"--- Extraindo áudio de {video_path} ---")
    try:
        os.makedirs(os.path.dirname(audio_output_path), exist_ok=True)
        video = VideoFileClip(video_path)
        audio = video.audio
        audio.write_audiofile(audio_output_path)
        audio.close()
        video.close()
        print("Áudio extraído com sucesso!\n")
        return True
    except Exception as e:
        print(f"Erro ao extrair áudio: {e}")
        return False

def upload_para_s3(s3_client, local_file, bucket, s3_key):
    print(f"--- Fazendo upload de {local_file} para S3 ---")
    try:
        s3_client.upload_file(local_file, bucket, s3_key, ExtraArgs={"ContentType": "audio/mpeg"})
        print("Upload realizado com sucesso!\n")
        return True
    except FileNotFoundError:
        print("Arquivo local não encontrado.")
    except NoCredentialsError:
        print("Credenciais AWS inválidas.")
    except ClientError as e:
        print(f"Erro AWS no upload: {e}")
    return False

def ejecutar_transcricao(transcribe_client, bucket, s3_audio_key, output_prefix):
    print("--- Iniciando AWS Transcribe ---")
    job_name = f"transcricao-{uuid.uuid4()}"
    input_file_uri = f"s3://{bucket}/{s3_audio_key}"
    
    try:
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": input_file_uri},
            MediaFormat="mp3",
            LanguageCode="pt-BR",
            OutputBucketName=bucket,
            OutputKey=output_prefix
        )
        print(f"Job '{job_name}' iniciado. Aguardando processamento...")

        while True:
            status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
            job_status = status["TranscriptionJob"]["TranscriptionJobStatus"]
            print(f"Status atual: {job_status}")
            if job_status in ["COMPLETED", "FAILED"]:
                break
            time.sleep(5)

        if job_status == "COMPLETED":
            print("Transcrição concluída com sucesso!\n")
            return f"{output_prefix}{job_name}.json"
        else:
            print(f"Falha no Job de Transcrição: {status}")
            return None
    except ClientError as e:
        print(f"Erro no Transcribe: {e}")
        return None

def analisar_sentimento_transcricao(s3_client, comprehend_client, bucket, json_key):
    print("--- Iniciando AWS Comprehend ---")
    try:
        response = s3_client.get_object(Bucket=bucket, Key=json_key)
        content = response["Body"].read().decode("utf-8")
        data = json.loads(content)
        texto = data["results"]["transcripts"][0]["transcript"]
        
        sentiment_response = comprehend_client.detect_sentiment(Text=texto, LanguageCode="pt")
        return {
            "texto": texto,
            "sentimento": sentiment_response['Sentiment'],
            "scores": sentiment_response['SentimentScore']
        }
    except Exception as e:
        print(f"Erro no Comprehend: {e}")
        return None
