from .utils import ocr_image, extract_summary_with_ner, save_file, handle_uploaded_file
from django.shortcuts import render
from django.http import HttpResponse, HttpRequest
from .forms import PDFUploadForm


def pdf_upload(request: HttpRequest) -> HttpResponse:
    form = PDFUploadForm()
    if request.method == 'POST':
        form = PDFUploadForm(request.POST, request.FILES)
        if form.is_valid():

            pdf_file = request.FILES['pdf_file']
            text = handle_uploaded_file(pdf_file)
            save_file(text, pdf_file.name)
            summary = extract_summary_with_ner(text)
            print(summary)

            context = {
                'form': form,
                'summary': summary,
            }
            return render(request, 'documents/index.html', context=context)

            # response = HttpResponse(doc_stream,
            #                         content_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
            # response['Content-Disposition'] = 'attachment; filename=summary.docx'
            # return response

    return render(request, 'documents/index.html', {'form': form})
