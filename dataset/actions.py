"""
Actions
========

This module consists of generalised admin actions such as export to csv, frequent
item sets.
"""

import csv
import json
import logging
from math import ceil

from django.http import HttpResponse

logger = logging.getLogger(__name__)


def dataset_division(ModelAdmin, request, queryset):
    """
    DATASET DIVISION
    =================

    Divides the given queryset to two parts (80/20) for ML training methods
    returns a tuple
    """
    training = ceil((queryset.count() * 80) / 100)
    evaluation = queryset.count() - training
    return (training, evaluation)


def prepare_training_set(ModelAdmin, request, queryset):
    """
    Exports the selection to CSV format -- dataset suitable for tensorflow training
    """

    headers = ["drug_1", "drug_2", "event", "prr", "drug1_prr", "drug2_prr", "predicted_value"]

    # divs = dataset_division(ModelAdmin, request, queryset)

    with open('export_data/trainingset2.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        # writer.writeheader()
        for data in queryset:

            row = {
                "drug_1": data.drug_1.id,
                "drug_2": data.drug_2.id,
                "event": data.event.id,
                "prr": data.prr,
                "drug1_prr" : data.drug1_prr,
                "drug2_prr" : data.drug2_prr,
                "predicted_value": data.predicted_value,}
            writer.writerow(row)
