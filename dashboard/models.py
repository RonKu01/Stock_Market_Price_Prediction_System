from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

class Stock(models.Model):

    STOCK_CHOICES = [
        ('GOOGL', 'Google'),
        ('AAPL', 'Apple'),
        ('MSFT', 'Microsoft'),
    ]

    MODEL_CHOICES = [
        ('voting_model', 'Voting'),
        ('svr_model', 'SVR'),
        ('rfr_model', 'RFR'),
        ('lstm_model', 'LSTM'),
    ]

    name = models.CharField(
        max_length=10, choices=STOCK_CHOICES, default='Google')
    
    ml_model = models.CharField(max_length=15, choices=MODEL_CHOICES, default='Voting')

    num_of_days = models.IntegerField(
        default=10,
        validators=[
            MinValueValidator(10),
            MaxValueValidator(20)
        ]
    )
    
    
    
