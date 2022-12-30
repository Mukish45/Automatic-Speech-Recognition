# Automatic Speech Recognition Model
  Whisper is an automatic speech recognition (ASR) system trained on 680,000 hours of multilingual and multitask supervised data collected from the web. It can be fine tuned for recognising others languages. SO, I have fine-tuned it to recognise Hindi speech and the model output the Hindi text. 

# Requirements
    !pip install git+https://github.com/huggingface/transformers
    !pip install gradio
    
# Code
    from transformers import pipeline
    import gradio as gr

    pipe = pipeline(model="mukish45/whisper-base-hi") 

    def transcribe(audio):
        text = pipe(audio)["text"]
        return text

    iface = gr.Interface(
        fn=transcribe, 
        inputs=gr.Audio(source="microphone", type="filepath"), 
        outputs="text",
        title="Whisper Base Hindi",
        description="Realtime demo for Hindi speech recognition using a fine-tuned Whisper base model.",
    )

    iface.launch()
    
# Output
