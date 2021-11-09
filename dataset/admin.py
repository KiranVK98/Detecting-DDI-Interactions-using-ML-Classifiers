from django.contrib import admin

from .models import Drug, Event, DrugToDrugInteraction
from .actions import prepare_training_set as PTS

class DrugAdmin(admin.ModelAdmin):
    list_display = ("drug_id", "drug_name")


class EventAdmin(admin.ModelAdmin):
    list_display = ("event_id", "event_name")

class DrugToDrugInteractionAdmin(admin.ModelAdmin):
    list_display = (
        "drug_1",
        "drug_2",
        "event",
        "prr",
        "drug1_prr",
        "drug2_prr",
        "predicted_value")

    actions = ["export_to_csv",]

    def export_to_csv(self, request, queryset):
        PTS(self, request, queryset)

    export_to_csv.short_description = "Export as CSV"


admin.site.register(Drug, DrugAdmin)
admin.site.register(Event, EventAdmin)
admin.site.register(DrugToDrugInteraction, DrugToDrugInteractionAdmin)