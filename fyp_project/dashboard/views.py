from django.shortcuts import render
from .forms import StockForm

def dashboard(request):
    SelectedStock = None
    form = StockForm()

    if request.method == "POST":
        form = StockForm(request.POST)

        if form.is_valid():
            SelectedStock = form.cleaned_data['name']

        return render(request, 'main/home.html', SelectedStock)

    context = {'form': form, 'SelectedStock': SelectedStock}
    return render(request, 'dashboard/dashboard.html', context)

