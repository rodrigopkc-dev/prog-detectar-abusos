import os
import cv2
import time
from ultralytics import YOLO

def analisar_video_yolo(video_path, model_path=r'D:\\notebook2\\python\\yolov8_facial_FAST_20260508_2217_medio\\weights\\best.pt'):
    """
    Roda a detecção de expressões faciais frame a frame usando OpenCV e YOLO.
    Retorna um dicionário estruturado com os resultados obtidos e gera um arquivo .txt de log.
    """
    print(f"--- Iniciando Análise YOLO no {video_path} ---")
    
    if not os.path.exists(video_path):
        return {
            "status": "ERRO",
            "erro": f"Arquivo de vídeo '{video_path}' não foi encontrado."
        }
        
    try:
        model = YOLO(model_path)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {
                "status": "ERRO",
                "erro": f"Não foi possível abrir o vídeo em {video_path}"
            }

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames = 1  # Analisar todos os frames (sem perdas)

        # Dicionários e listas para coleta
        estatisticas = {label: 0 for label in model.names.values()}
        alertas_criticos = []
        sentimentos_alerta = ['Fear', 'Sad', 'Angry', 'Disgust']

        print(f"Processando frames com YOLO de Alta Sensibilidade...")
        inicio_proc = time.time()
        frame_idx = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frame_idx % skip_frames == 0:
                results = model.predict(frame, conf=0.25, verbose=False) 
                
                timestamp = frame_idx / fps
                minutos = int(timestamp // 60)
                segundos = int(timestamp % 60)

                for r in results:
                    for box in r.boxes:
                        label = model.names[int(box.cls[0])]
                        estatisticas[label] += 1
                        
                        if label in sentimentos_alerta:
                            alertas_criticos.append(f"[{minutos:02d}:{segundos:02d}] - {label}")

            frame_idx += 1

        cap.release()
        fim_proc = time.time()
        tempo_total_video = frame_idx / fps

        # Métricas de Cálculo de Risco
        score_risco = sum(estatisticas[s] for s in sentimentos_alerta)
        segundos_em_alerta = len(set([a.split("]")[0].replace("[", "") for a in alertas_criticos]))
        taxa_presenca_alerta = (segundos_em_alerta / tempo_total_video) * 100 if tempo_total_video > 0 else 0

        # Definição do Veredito
        if score_risco >= 30 or taxa_presenca_alerta > 10:
            veredito = "[CRÍTICO] ALTO RISCO - Padrões de sofrimento/violência detectados."
        elif score_risco > 5 or taxa_presenca_alerta > 2:
            veredito = "[ATENÇÃO] RISCO MODERADO - Sinais isolados ou microexpressões de alerta."
        else:
            veredito = "[NORMALIDADE] Sem sinais expressivos de desconforto."

        # --- NOVA PARTE: GERAR ARQUIVO DE LOG COMPLETO ---
        alertas_limpos = list(dict.fromkeys(alertas_criticos)) # Remove duplicados exatos no mesmo segundo
        nome_base = os.path.splitext(os.path.basename(video_path))[0]
        arquivo_log = f"log_{nome_base}.txt"
        
        with open(arquivo_log, "w", encoding="utf-8") as f:
            f.write(f"=== LOG DE MOMENTOS CRÍTICOS ===\n")
            f.write(f"Vídeo analisado: {video_path}\n")
            f.write(f"Data/Hora do processamento: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total de alertas gerados: {len(alertas_limpos)}\n")
            f.write(f"Veredito do YOLO: {veredito}\n")
            f.write("="*40 + "\n\n")
            
            if not alertas_limpos:
                f.write("Nenhum sinal de alerta detectado.\n")
            else:
                for alerta in alertas_limpos:
                    f.write(f"⚠ {alerta}\n")
                    
        print(f"Arquivo de log completo gerado com sucesso: '{arquivo_log}'")
        print("Análise YOLO concluída com sucesso!\n")
        
        # Retorna os dados estruturados para o script principal
        return {
            "status": "SUCESSO",
            "tempo_processamento": int(fim_proc - inicio_proc),
            "estatisticas": estatisticas,
            "alertas_criticos": alertas_criticos,
            "score_risco": score_risco,
            "taxa_presenca_alerta": taxa_presenca_alerta,
            "veredito": veredito,
            "arquivo_log": arquivo_log # Passa o nome do arquivo gerado para o principal se quiser usar
        }
        
    except Exception as e:
        print(f"Erro na análise do YOLO: {e}")
        return {
            "status": "ERRO",
            "erro": str(e)
        }