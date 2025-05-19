from django.shortcuts import render
from summarizer.summarizer_logic import textrank, get_sentence_vector # Import all necessary functions

def summarize_view(request):
    if request.method == "POST":
        text_to_summarize = request.POST.get('text')
        extractive_summary = textrank(text_to_summarize.split('।'), embeddings)
        summary = '।'.join(extractive_summary)
        return render(request, 'summarizer/result.html', {'summary': summary})
    
    return render(request, 'summarizer/form.html')