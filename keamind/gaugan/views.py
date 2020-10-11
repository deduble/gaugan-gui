from django.shortcuts import render

# Create your views here.
def gaugan(request):
    return render(request, "gaugan/index.html")