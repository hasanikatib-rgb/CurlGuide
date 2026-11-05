"""CurlGuide: experimental hair pattern classifier. Run: python app.py"""
from pathlib import Path
import os
import numpy as np
import gradio as gr
import joblib
from PIL import Image, ImageOps
from train_features import features

ROOT = Path(__file__).resolve().parent
MODEL = joblib.load(ROOT / 'hair_model.joblib')
TYPES = [f'{n}{letter}' for n in range(1,5) for letter in 'ABC']
CARE = {
 '1': {'name':'Straight', 'wash':'Use a gentle, lightweight shampoo as needed; condition mainly the mid-lengths and ends.', 'style':'Try a light leave-in spray for softness and a little mousse for volume.', 'product':'Lightweight conditioner, leave-in mist, light mousse.', 'avoid':'Too many heavy oils or rich creams near the roots.'},
 '2': {'name':'Wavy', 'wash':'Use a gentle shampoo and conditioner; detangle gently while damp.', 'style':'Scrunch a lightweight mousse or gel into damp hair, then air-dry or diffuse on low.', 'product':'Light mousse, flexible-hold gel, lightweight leave-in.', 'avoid':'Rough towel drying or repeated touching while drying.'},
 '3': {'name':'Curly', 'wash':'Condition thoroughly and detangle in sections with a wide-tooth comb while damp.', 'style':'Apply leave-in and curl gel to damp hair; gently scrunch and let curls set.', 'product':'Moisturizing conditioner, curl gel, optional curl cream.', 'avoid':'Dry brushing if it makes curls frizzy; excessive heat.'},
 '4': {'name':'Very Curly (4A–4C)', 'wash':'Cleanse gently and condition in sections; take your time detangling with plenty of slip.', 'style':'Work in sections with a moisturizing leave-in and gel or cream; try low-tension protective styles.', 'product':'Rich conditioner, leave-in, moisturizing cream or gel.', 'avoid':'Tight styles that pull on the scalp or aggressive detangling.'}
}

def predict(photo):
    if photo is None:
        return '### Upload a photo to start', 'Choose a clear, well-lit photo showing your natural hair pattern.', '', ''
    try:
        im = ImageOps.exif_transpose(photo).convert('RGB')
        if min(im.size) < 80:
            return '### Photo too small', 'Please choose a clearer photo at least 80 × 80 pixels.', '', ''
        scores = np.asarray(MODEL.predict_proba([features(im)])[0], dtype=float)
        order = np.argsort(scores)[::-1]
        best = str(MODEL.classes_[order[0]]).upper()
        first = float(scores[order[0]])
        second = float(scores[order[1]])
        uncertain = first < 0.45 or (first - second) < 0.15
        family = CARE[best[0]]
        if uncertain:
            headline = '### Pattern unclear — more than one type is possible'
            note = f'The model leaned toward **{best}**, but this image does not support a confident subtype estimate. Try a brighter photo with your natural pattern clearly visible.'
        else:
            headline = f'### Possible hair type: {best}'
            note = f'**{family["name"]} hair family.** This is a tentative estimate, not a verified classification.'
        alternatives = '\n'.join(f'- **{str(MODEL.classes_[i]).upper()}** — {scores[i]*100:.0f}% model score' for i in order[:3])
        result = f'{headline}\n\n{note}\n\n**Closest model matches**\n{alternatives}\n\n*Model scores are not calibrated probabilities.*'
        advice = (f'### Care ideas for {family["name"].lower()} patterns\n\n'
                  f'**Wash & condition:** {family["wash"]}\n\n'
                  f'**Style:** {family["style"]}\n\n'
                  f'**Product types to try:** {family["product"]}\n\n'
                  f'**Be mindful of:** {family["avoid"]}')
        steps = '1. Take a photo in natural light with your natural, dry hair pattern visible.\n2. Try a second angle if the result is uncertain.\n3. Use the suggestions as starting points and adjust to how your hair responds.'
        warning = ('**Research preview, not a reliable diagnosis.** The current nine-class model scored **17.8%** on its held-out test set. '
                   'The Very Curly (4A–4C) category is not yet detectable by this model. If your hair is coily, this model may label it incorrectly; do not treat that label as fact. '
                   'A photo cannot establish porosity, scalp health, or exact product needs.')
        return result, advice, steps, warning
    except Exception:
        return '### Could not analyze this photo', 'Try a standard JPG or PNG with the hair clearly visible.', '', ''

CSS = '''
.gradio-container {max-width:1120px!important;margin:auto!important; font-family:Inter,ui-sans-serif,system-ui!important}
#hero {background:linear-gradient(120deg,#102b28,#285f55);border-radius:22px;padding:30px 34px;color:white;margin-bottom:18px}
#hero h1,#hero p {color:white!important}
#hero h1 {font-size:38px!important;letter-spacing:-1.2px}
#hero p {opacity:.89}
#analysis {border:1px solid #d7e5df;border-radius:16px;padding:18px!important}
#action {background:#166b58!important;color:white!important;border-radius:12px!important;font-weight:700!important}
#notice {border:1px solid #eedca8;background:#fff9e9;border-radius:12px;padding:14px}
'''
with gr.Blocks(title='CurlGuide | AI Hair Pattern Explorer', css=CSS, theme=gr.themes.Soft(primary_hue='emerald')) as demo:
    gr.HTML('''<div id="hero"><div style="font-size:13px;font-weight:750;letter-spacing:2px;opacity:.8">CURLGUIDE / AI HAIR LAB</div><h1>Understand your hair pattern.</h1><p>Upload a photo. Explore a tentative hair-type estimate and a care routine made for that pattern — no manual hair-type selection.</p></div>''')
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            photo = gr.Image(type='pil', sources=['upload','webcam'], label='Your hair photo', height=340)
            btn = gr.Button('Analyze my hair →', variant='primary', elem_id='action', size='lg')
            gr.Markdown('**Photo tips:** Natural lighting · Hair in focus · Show the pattern, not just the scalp · Avoid filters')
        with gr.Column(scale=6, elem_id='analysis'):
            result = gr.Markdown('### Ready when you are\nUpload a photo and select **Analyze my hair**.')
            advice = gr.Markdown('Your personalized care ideas will appear here.')
    with gr.Accordion('How to get a better result', open=False):
        steps = gr.Markdown('Take a clear photo of your natural hair pattern from the side or back.')
    with gr.Accordion('Hair type guide and limitations', open=True):
        gr.Markdown('**Straight:** 1A–1C · **Wavy:** 2A–2C · **Curly:** 3A–3C · **Very Curly:** 4A–4C. The numbers describe broad curl patterns, not hair health or quality. Patterns can overlap on one head.')
        warning = gr.Markdown('**Experimental model:** Trained on 1A–3C only. Its held-out test accuracy is 17.8%; predictions are often wrong. Very Curly (4A–4C) is shown in the guide but is not yet recognized by the model.', elem_id='notice')
    gr.Markdown('**Privacy:** When run locally, photos stay on your machine for processing. If hosted online, photos are sent to the server running the app. No photos are saved by this app code.')
    btn.click(predict, inputs=[photo], outputs=[result, advice, steps, warning])
    photo.change(predict, inputs=[photo], outputs=[result, advice, steps, warning])

if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")), share=False)
