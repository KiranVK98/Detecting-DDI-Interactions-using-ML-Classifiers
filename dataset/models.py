from django.db import models
import numpy as np
from .tensor_predictor import get_linear

class Drug(models.Model):
    drug_id = models.CharField(max_length=12, unique=True)
    drug_name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.drug_name

class Event(models.Model):
    event_id = models.CharField(verbose_name="Event ULMS ID", max_length=8, unique=True)
    event_name = models.CharField(max_length=50, unique=True)

    def __str__(self):
        return self.event_name


class DrugToDrugInteraction(models.Model):
    drug_1 = models.ForeignKey(Drug, related_name="Drug_1", on_delete=models.CASCADE)
    drug_2 = models.ForeignKey(Drug, related_name="Drug_2", on_delete=models.CASCADE)
    event = models.ForeignKey(Event, models.CASCADE)

    prr = models.FloatField(default=0.00, verbose_name="Proportional Reporting Ratio")
    drug1_prr = models.FloatField(default=0.00, verbose_name="Drug 1 PRR")
    drug2_prr = models.FloatField(default=0.00, verbose_name="Drug 2 PRR")

    theoritical_value = models.FloatField(default=0.00)
    predicted_value = models.FloatField(default=0.00)

    def clean(self):
        input_dict = {
            "drug_1": np.array([float(self.drug_1.id),]),
            "drug_2": np.array([float(self.drug_2.id),]),
            "event": np.array([float(self.event.id),]),
            "prr": np.array([self.prr,]),
            "drug1_prr": np.array([self.drug1_prr,]),
            "drug2_prr": np.array([self.drug2_prr,]),
            }

        self.predicted_value = get_linear(input_dict)[0]

    def __str__(self):
        return "{}::{}::{}".format(self.drug_1, self.drug_2, self.event)