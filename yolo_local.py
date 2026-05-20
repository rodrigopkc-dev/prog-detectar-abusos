import os
import cv2
import time
from ultralytics import YOLO

def analisar_video_yolo(video_path, model_path=r'D:\\notebook2\\python\\yolov8_facial_FAST_20260508_2217_medio\\weights\\best.pt'):
    """
    Roda a detecção de expressões faciais focado exclusivamente em Fear, Sad, Angry e Disgust.
    Ajustado com maior tolerância para evitar alarmes falsos em frames isolados.
    """
    print(f"--- Iniciando Análise de Risco YOLO no {video_path} ---")
    
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
        skip_frames = 1  # Analisa todos os frames para manter a precisão temporal

        # Dicionários e listas para coleta
        estatisticas = {label: 0 for label in model.names.values()}
        alertas_criticos = []
        
        # FOCO EXCLUSIVO: Padrões de sofrimento, violência e desconforto
        #sentimentos_alerta = ['Fear', 'Sad', 'Angry', 'Disgust']
        sentimentos_alerta = ['Fear', 'Sad', 'Angry']

        print(f"Monitorando padrões de desconforto com filtros calibrados...")
        inicio_proc = time.time()
        frame_idx = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frame_idx % skip_frames == 0:
                # AJUSTE 1: Subimos para 0.42. Corta falsos positivos cruciais
                # onde o modelo "chuta" que um rosto neutro está triste ou bravo.
                results = model.predict(frame, conf=0.42, verbose=False) 
                
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

        # Cálculos de Indicadores de Sofrimento
        score_risco = sum(estatisticas[s] for s in sentimentos_alerta)
        
        # Identifica quantos segundos únicos do vídeo contiveram alertas
        segundos_em_alerta = len(set([a.split("]")[0].replace("[", "") for a in alertas_criticos]))
        taxa_presenca_alerta = (segundos_em_alerta / tempo_total_video) * 100 if tempo_total_video > 0 else 0

        # =======================================================
        # LÓGICA DE VEREDITO EQUILIBRADA (Sem ser severa demais)
        # =======================================================
        # AJUSTE 2: Agora exige um volume real de frames ou uma presença persistente no vídeo
        if score_risco >= 90 or taxa_presenca_alerta > 18:
            veredito = "[CRÍTICO] ALTO RISCO - Padrões frequentes e persistentes de sofrimento/desconforto detectados."
        elif score_risco > 25 or taxa_presenca_alerta > 4:
            veredito = "[ATENÇÃO] RISCO MODERADO - Sinais isolados ou flutuações emocionais negativas pontuais."
        else:
            veredito = "[NORMALIDADE] Sem sinais expressivos ou contínuos de desconforto."

        # --- GERAR ARQUIVO DE LOG DE AUDITORIA ---
        alertas_limpos = list(dict.fromkeys(alertas_criticos)) 
        nome_base = os.path.splitext(os.path.basename(video_path))[0]
        arquivo_log = f"log_risco_{nome_base}.txt"
        
        with open(arquivo_log, "w", encoding="utf-8") as f:
            f.write(f"=== ANÁLISE DE COMPORTAMENTO CRÍTICO ===\n")
            f.write(f"Arquivo analisado: {video_path}\n")
            f.write(f"Data do disparo: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total de frames de desconforto acumulados: {score_risco}\n")
            f.write(f"Presença de tensão na linha do tempo: {taxa_presenca_alerta:.1f}%\n")
            f.write(f"VEREDITO FINAL: {veredito}\n")
            f.write("="*50 + "\n\n")
            
            if not alertas_limpos:
                f.write("Nenhum padrão de sofrimento expressivo foi mapeado.\n")
            else:
                f.write("Log cronológico das ocorrências:\n")
                for alerta in alertas_limpos:
                    f.write(f" ⚠ {alerta}\n")
                    
        print(f"Log gerado com sucesso: '{arquivo_log}'")
        
        return {
            "status": "SUCESSO",
            "tempo_processamento": int(fim_proc - inicio_proc),
            "estatisticas": estatisticas,
            "alertas_criticos": alertas_criticos,
            "score_risco": score_risco,
            "taxa_presenca_alerta": taxa_presenca_alerta,
            "veredito": veredito,
            "arquivo_log": arquivo_log
        }
        
    except Exception as e:
        print(f"Erro na análise do YOLO: {e}")
        return {
            "status": "ERRO",
            "erro": str(e)
        }