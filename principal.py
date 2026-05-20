import tempfile
import streamlit as str
import aws_codigo_completo as aws
import yolo_local as yolo
import streamlit as st

# Configuração inicial da página web
st.set_page_config(page_title="Relatório Consolidado de IA", layout="wide")

st.title("🎬 Analisador de Vídeo com IA")
st.subheader("Processamento de Áudio (AWS) & Visão Computacional (YOLO)")

# ==========================================
# SELEÇÃO DO ARQUIVO (UPLOAD)
# ==========================================
# Permite escolher qualquer arquivo mp4 do computador
video_upload = st.file_uploader("Escolha um arquivo de vídeo do seu PC", type=["mp4"])

if video_upload is not None:
    # O Streamlit carrega o arquivo na memória. Precisamos salvar em um arquivo temporário 
    # no disco para que as funções da AWS e do YOLO consigam ler o caminho do arquivo.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(video_upload.read())
        video_origem = tfile.name

    st.success("Vídeo carregado com sucesso! Pronto para processar.")
    
    # Botão para iniciar o fluxo
    if st.button("▶ Iniciar Processamento", type="primary"):
        
        # Estruturas para armazenar o resumo
        relatorio_aws = {"status": "Não executado", "texto": "", "sentimento": "", "scores": {}}
        relatorio_yolo = {"status": "Não executado", "dados": {}}
        
        # Cria um aviso de carregamento na tela
        with st.spinner("Processando vídeo... Por favor, aguarde."):
            
            # 0. Inicializa os clientes da AWS
            s3_cli, transcribe_cli, comprehend_cli = aws.obter_clientes_aws()
            
            # -------------------------------------------------------
            # FLUXO 1: PROCESSAMENTO AWS (Áudio & Texto)
            # -------------------------------------------------------
            if aws.extrair_audio(video_origem, aws.LOCAL_AUDIO_PATH):
                if aws.upload_para_s3(s3_cli, aws.LOCAL_AUDIO_PATH, aws.BUCKET_NAME, aws.S3_AUDIO_KEY):
                    
                    json_s3_key = aws.ejecutar_transcricao(
                        transcribe_cli, 
                        aws.BUCKET_NAME, 
                        aws.S3_AUDIO_KEY, 
                        aws.OUTPUT_KEY_PREFIX
                    )
                    
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
        # EXIBIÇÃO CONSOLIDADA NA TELA DO STREAMLIT
        # =======================================================
        st.write("---")
        st.header("📊 Relatório Consolidado")
        
        # Divide a tela em duas colunas (AWS na esquerda, YOLO na direita)
        col1, col2 = st.columns(2)
        
        # --- Coluna 1: Resultados AWS ---
        with col1:
            st.subheader("☁️ Resultado AWS")
            
            if relatorio_aws["status"] == "SUCESSO":
                st.success(f"Status: {relatorio_aws['status']}")
                st.info(f"**Texto Transcrito:**\n\"{relatorio_aws['texto']}\"")
                st.metric(label="Sentimento Geral", value=relatorio_aws['sentimento'])
                
                st.write("**Scores Detalhados:**")
                for sentimento, valor in relatorio_aws["scores"].items():
                    # Transforma o score em barra de progresso (0.0 a 1.0)
                    st.write(f"{sentimento}: {valor:.4f}")
                    st.progress(float(valor))
            else:
                st.error(f"Status do Pipeline: {relatorio_aws['status']}")
                
        # --- Coluna 2: Resultados YOLO Local ---
        with col2:
            st.subheader("👁️ Resultado YOLO Local")
            
            if relatorio_yolo["status"] == "SUCESSO":
                st.success(f"Status: {relatorio_yolo['status']}")
                dados = relatorio_yolo["dados"]
                
                # Exibe métricas em blocos destacados
                m1, m2, m3 = st.columns(3)
                m1.metric("Tempo Processamento", f"{dados['tempo_processamento']}s")
                m2.metric("Presença de Alerta", f"{dados['taxa_presenca_alerta']:.1f}%")
                m3.metric("Gatilhos / Risco", dados['score_risco'])
                
                st.warning(f"**Veredito Final:** {dados['veredito']}")
                
                st.write("**Distribuição de Sentimentos:**")
                for sentimento, total in dados["estatisticas"].items():
                    st.text(f"{sentimento:10} | {total:4} ocorrências")
                    
                st.write("**Log de Momentos Críticos (Top 5):**")
                alertas_limpos = list(dict.fromkeys(dados["alertas_criticos"]))
                if not alertas_limpos:
                    st.write("Nenhum sinal de alerta detectado.")
                else:
                    for a in alertas_limpos[:5]:
                        st.caption(f"⚠️ {a}")
                    if len(alertas_limpos) > 5:
                        st.caption(f"... (+ {len(alertas_limpos) - 5} alertas na lista)")
                    
                    st.info(f"O log completo foi salvo em: `{dados['arquivo_log']}`")
            else:
                st.error(f"Status da Análise: {relatorio_yolo['status']}")

else:
    st.info("Por favor, selecione um arquivo de vídeo para começar.")
