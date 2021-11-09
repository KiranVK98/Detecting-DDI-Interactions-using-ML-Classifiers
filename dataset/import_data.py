import csv
from dataset.models import Drug, Event, DrugToDrugInteraction

def get_d(d_id):
    try:
        return Drug.objects.get(drug_id=d_id)
    except:
        return None


def get_e(e_id):
    try:
        return Event.objects.get(event_id=e_id)
    except:
        return None



filename = "raw_data/d-d-i.csv"

with open(filename, "rt") as c_file:
    data = csv.reader(c_file)


with open(filename, "rt") as c_file:
    data = csv.reader(c_file)
    for row in data:
        drug_1 = get_d(row[0])
        drug_2 = get_d(row[1])
        event = get_e(row[2])
        prr = row[3]
        drug1_prr = row[4]
        drug2_prr = row[5]
        value = row[6]
        print(drug_1, drug_2, event)
        new_ddi = DrugToDrugInteraction(
            drug_1=drug_1,
            drug_2=drug_2,
            event=event,
            drug1_prr=drug1_prr,
            drug2_prr=drug2_prr,
            predicted_value=value)
        try:
            new_ddi.save()
        except Exception as e:
            print(row[0], row[1], e)

