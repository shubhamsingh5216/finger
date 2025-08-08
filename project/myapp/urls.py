from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import AddFingerprintView, MatchFingerprintView

urlpatterns = [
    path('add/', AddFingerprintView.as_view(), name='add_fingerprint'),
    path('match/', MatchFingerprintView.as_view(), name='match_fingerprint'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
