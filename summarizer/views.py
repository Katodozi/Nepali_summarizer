from summarizer.summarizer_logic import textrank, embeddings
from django.shortcuts import render

def index_view(request):#this function handles everything in the root url page
    return render(request, "summarizer/index.html")

#this function handles all the request related to the summarizer page and result page for which there are certain conditions to be satisfied
def summarize_view(request):
    if request.method == "POST":#user input text messege and submit using post method which is checked here
        text_to_summarize = request.POST.get('text')#the text that user entered is retrieved here
        
        # Checking for empty input
        if not text_to_summarize:
            return render(request, "summarizer/form.html", {'error': 'Please enter some text to summarize.'})
        
        sentences = text_to_summarize.split('।')

        if len(sentences)<6:
            return render(request, "summarizer/form.html", {'error': 'Please Enter at least 5 sentences to summarize..!!'} )
        elif len(sentences)>35:
            return render(request, "summarizer/form.html", {'error': 'Please Enter less 35 sentences at once ..!!'} )
        #Generating extractive summary
        extractive_summary = textrank(text_to_summarize.split('।'), embeddings)
        
        # Joining the summary into a single string
        summary = '।'.join(extractive_summary)

        return render(request, "summarizer/result.html", {'summary': summary})#if the text is entered then it return the summary in the result page
    
    return render(request, "summarizer/form.html")