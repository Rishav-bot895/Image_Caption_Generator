from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import tempfile
import os
import io
from PIL import Image
from utils.test import ImageCaptionGenerator

# Initialize once on startup
try:
    MODEL_PATH = 'C:/Users/Raj/Desktop/Projects/Image Caption Generator/WebApp/webapp/utils/best_model.keras'
    TOKENIZER_PATH = 'C:/Users/Raj/Desktop/Projects/Image Caption Generator/WebApp/webapp/utils/tokenizer.pkl'
    caption_generator = ImageCaptionGenerator(MODEL_PATH, TOKENIZER_PATH)
    print("Caption generator ready.")
except Exception as e:
    print(f"Initialization error: {e}")
    caption_generator = None

def home(request):
    if request.method == 'GET':
        return render(request, 'home.html')
    if request.method == 'POST':
        if caption_generator is None:
            return JsonResponse({'success': False, 'error': 'Model not initialized.'})
        if 'image' not in request.FILES:
            return JsonResponse({'success': False, 'error': 'No file uploaded.'})
        image_file = request.FILES['image']
        if not image_file.content_type.startswith('image/'):
            return JsonResponse({'success': False, 'error': 'Upload an image.'})
        if image_file.size > 10 * 1024 * 1024:
            return JsonResponse({'success': False, 'error': 'Max size is 10MB.'})

        # Save to temp
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        try:
            tmp.write(image_file.read())
            tmp.close()
            caption = caption_generator.generate_caption(tmp.name, display_image=False)
            if caption:
                return JsonResponse({
                    'success': True,
                    'caption': caption,
                    'filename': image_file.name,
                    'filesize': f"{image_file.size / (1024*1024):.2f} MB"
                })
            else:
                return JsonResponse({'success': False, 'error': 'Caption failed.'})
        finally:
            try: os.unlink(tmp.name)
            except: pass

@csrf_exempt
def generate_caption_ajax(request):
    return home(request) if request.method == 'POST' else JsonResponse({'success': False, 'error': 'Invalid method'})

def clear_buffer(request):
    if request.method == 'POST':
        return JsonResponse({'success': True, 'message': 'Nothing to clear.'})
    return JsonResponse({'success': False, 'error': 'Invalid method'})
