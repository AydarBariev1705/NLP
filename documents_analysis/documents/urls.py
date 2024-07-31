from django.urls import path
from django.conf import settings
from .views import pdf_upload
from django.conf.urls.static import static
app_name = 'get_weather'

urlpatterns = [
    path('', pdf_upload, name='docs'),
]
urlpatterns.extend(
        static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    )
urlpatterns.extend(
        static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    )
