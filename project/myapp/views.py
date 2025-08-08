import os
from django.core.files import File
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from .models import Fingerprint
from .utils import match_fingerprint

def handle_uploaded_file(file):
    temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.name)
    with open(temp_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    return temp_path

class AddFingerprintView(APIView):
    def post(self, request):
        name = request.data.get('name')
        image_file = request.FILES.get('image')
        image_path = request.data.get('image_path')

        if not name or (not image_file and not image_path):
            return Response({"error": "Missing name or image"}, status=status.HTTP_400_BAD_REQUEST)

        if image_path and not image_file:
            abs_path = os.path.join(settings.MEDIA_ROOT, image_path)
            if not os.path.exists(abs_path):
                return Response({"error": "Image path does not exist"}, status=status.HTTP_400_BAD_REQUEST)
            with open(abs_path, 'rb') as f:
                image_file = File(f, name=os.path.basename(abs_path))
                fingerprint = Fingerprint.objects.create(name=name, image=image_file)
        else:
            fingerprint = Fingerprint.objects.create(name=name, image=image_file)

        return Response({"message": "Fingerprint added", "id": fingerprint.id}, status=status.HTTP_201_CREATED)

@method_decorator(csrf_exempt, name='dispatch')
class MatchFingerprintView(APIView):
    def post(self, request):
        image_file = request.FILES.get('image')
        image_path = request.data.get('image_path')

        if not image_file and not image_path:
            return Response({"error": "Missing image or image_path"}, status=status.HTTP_400_BAD_REQUEST)

        if image_path and not image_file:
            abs_path = os.path.join(settings.MEDIA_ROOT, image_path)
            if not os.path.exists(abs_path):
                return Response({"error": "Image path does not exist"}, status=status.HTTP_400_BAD_REQUEST)
            temp_path = abs_path
        else:
            temp_path = handle_uploaded_file(image_file)

        best_score = 0
        best_match_name = None
        for fp in Fingerprint.objects.all():
            score = match_fingerprint(temp_path, fp.image.path)
            if score > best_score:
                best_score = score
                best_match_name = fp.name

        # Optionally delete temp file if uploaded
        if image_file and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

        if best_score > 0.6:
            return Response({"match": True, "name": best_match_name, "score": best_score})
        else:
            return Response({"match": False, "score": best_score})
