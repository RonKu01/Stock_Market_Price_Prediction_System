from django import forms
from .models import Stock

class StockForm(forms.ModelForm):
    class Meta:
        model = Stock
        fields = ('name','ml_model', 'num_of_days')