import json
import aws_codigo_completo as aws
import yolo_local as yolo

# ==========================================
# FLUXO PRINCIPAL (EXECUÇÃO)
# ==========================================
if __name__ == "__main__":
    
    video_origem = "video//video01.mp4"
    
    # Estruturas para armazenar o resumo que será exibido no final
    relatorio_aws = {"status": "Não executado", "texto": "", "sentimento": "", "scores": {}}
    relatorio_yolo = {"status": "Não executado", "dados": {}}
    
    # 0. Inicializa os clientes da AWS
    s3_cli, transcribe_cli, comprehend_cli = aws.obter_clientes_aws()
    
    # -------------------------------------------------------
    # FLUXO 1: PROCESSAMENTO AWS (Áudio & Texto)
    # -------------------------------------------------------
    if aws.extrair_audio(video_origem, aws.LOCAL_AUDIO_PATH):
        if aws.upload_para_s3(s3_cli, aws.LOCAL_AUDIO_PATH, aws.BUCKET_NAME, aws.S3_AUDIO_KEY):
            
            # Executa a transcrição
            json_s3_key = aws.ejecutar_transcricao(
                transcribe_cli, 
                aws.BUCKET_NAME, 
                aws.S3_AUDIO_KEY, 
                aws.OUTPUT_KEY_PREFIX
            )
            
            # Se a transcrição deu certo, roda a análise de sentimento
            if json_s3_key:
                resultado_comprehend = aws.analisar_sentimento_transcricao(
                    s3_cli, comprehend_cli, aws.BUCKET_NAME, json_s3_key
                )
                
                if resultado_comprehend:
                    relatorio_aws["status"] = "SUCESSO"
                    relatorio_aws["texto"] = resultado_comprehend["texto"]
                    relatorio_aws["sentimento"] = resultado_comprehend["sentimento"]
                    relatorio_aws["scores"] = resultado_comprehend["scores"]
                else:
                    relatorio_aws["status"] = "FALHA NA ANÁLISE DE SENTIMENTO"
            else:
                relatorio_aws["status"] = "FALHA NA TRANSCRIÇÃO DE ÁUDIO"
        else:
            relatorio_aws["status"] = "FALHA NO UPLOAD PARA O S3"
    else:
        relatorio_aws["status"] = "FALHA NA EXTRAÇÃO DO ÁUDIO"


    # -------------------------------------------------------
    # FLUXO 2: PROCESSAMENTO LOCAL (YOLO Visão Computacional)
    # -------------------------------------------------------
    resultado_yolo_raw = yolo.analisar_video_yolo(video_origem)
    
    if resultado_yolo_raw["status"] == "SUCESSO":
        relatorio_yolo["status"] = "SUCESSO"
        relatorio_yolo["dados"] = resultado_yolo_raw
    else:
        relatorio_yolo["status"] = f"FALHA: {resultado_yolo_raw.get('erro')}"


    # =======================================================
    # EXIBIÇÃO CONSOLIDADA DOS RESULTADOS FINAIS
    # =======================================================
    print("\n" + "="*60)
    print("                RELATÓRIO CONSOLIDADO DE IA             ")
    print("="*60)
    
    # --- Bloco de Resultados AWS ---
    print("\n[ RESULTADO AWS ]")
    print(f"Status do Pipeline: {relatorio_aws['status']}")
    if relatorio_aws["status"] == "SUCESSO":
        print(f"Texto Transcrito:   \"{relatorio_aws['texto']}\"")
        print(f"Sentimento Geral:   {relatorio_aws['sentimento']}")
        print("Scores Detalhados:")
        for sentimento, valor in relatorio_aws["scores"].items():
            print(f"  - {sentimento}: {valor:.4f}")
    
    print("-" * 50)
    
    # --- Bloco de Resultados YOLO Local ---
    print("\n[ RESULTADO YOLO LOCAL (EXPRESSÕES FACIAIS) ]")
    print(f"Status da Análise: {relatorio_yolo['status']}")
    if relatorio_yolo["status"] == "SUCESSO":
        dados = relatorio_yolo["dados"]
        print(f"Tempo total de processamento: {dados['tempo_processamento']}s")
        print(f"Presença de adversidade no vídeo: {dados['taxa_presenca_alerta']:.1f}% do tempo")
        print(f"Total de gatilhos detectados: {dados['score_risco']}")
        print(f"Veredito Final: {dados['veredito']}")
        
        print("\nDistribuição de Sentimentos:")
        for sentimento, total in dados["estatisticas"].items():
            barra = "█" * (total // 10) if total > 0 else ""
            print(f"  {sentimento:10} | {total:4} ocorrências {barra}")
            
        print("\nLog de Momentos Críticos (Apenas primeiros 5 exemplos):")
        alertas_limpos = list(dict.fromkeys(dados["alertas_criticos"]))
        if not alertas_limpos:
            print("  Nenhum sinal de alerta detectado.")
        else:
            for a in alertas_limpos[:5]:
                print(f"  ⚠ {a}")
            if len(alertas_limpos) > 5:
                print(f"  ... (+ {len(alertas_limpos) - 5} alertas na lista)")
            
            # Avisa onde encontrar a lista completa
            print(f"\n  >> O log completo com todos os {len(alertas_limpos)} alertas foi salvo em: '{dados['arquivo_log']}'")
            
    print("\n" + "="*60)