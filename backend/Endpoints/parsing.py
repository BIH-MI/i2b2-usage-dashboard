import xml.etree.ElementTree as ET

def parse_query_xml(xml_string):
    root = ET.fromstring(xml_string)

    #invert lives on the panel, not the item, so concepts are split panel-by-panel
    #invert == '1' means excluded, otherwise included
    included_concepts = []
    excluded_concepts = [] 
    panels = [] #query structure as built in i2b2: AND between panels, OR between items of one panel
    for panel in root.findall('.//panel'):
        invert = panel.findtext('invert')
        keys = [el.text for el in panel.findall('./item/item_key')]
        if invert == '1':
            excluded_concepts.extend(keys)
        else:
            included_concepts.extend(keys)
        occ = int(panel.findtext('total_item_occurrences', '1'))

        items = []
        for item in panel.findall('./item'):
            date_el = item.find('constrain_by_date')
            value_el = item.find('constrain_by_value')
            items.append({
                "key": item.findtext('item_key'),
                "date_from": date_el.findtext('date_from') if date_el is not None else None,
                "date_to": date_el.findtext('date_to') if date_el is not None else None,
                "value_operator": value_el.findtext('value_operator') if value_el is not None else None,
                "value_constraint": value_el.findtext('value_constraint') if value_el is not None else None,
                "value_unit": value_el.findtext('value_unit_of_measure') if value_el is not None else None
            })
        panels.append({"excluded": invert == '1', "occurrences": occ, "items": items})
        

    item_keys = included_concepts + excluded_concepts

    #taken from panels, so the xml is only read once
    date_constraints_values = [
        {"date_from": i["date_from"], "date_to": i["date_to"]}
        for p in panels for i in p["items"] if i["date_from"] or i["date_to"]
    ]

    
    date_constraint = len(date_constraints_values)
    value_constraint = len(root.findall('.//item/constrain_by_value'))
    occ_constraints = sum(1 for p in panels if p["occurrences"] > 1)
    num_constraints = date_constraint + value_constraint + occ_constraints

    temporal = len(root.findall('.//subquery_constraint'))

    return {
        "item_keys": item_keys,
        "included_concepts": included_concepts,
        "excluded_concepts": excluded_concepts,
        "num_constraints": num_constraints,
        "date_constraints": date_constraints_values, #actual date_from/date_to values, one entry per constrain_by_date found
        "panels": panels,
        "temporal_query_constraints": temporal,

    }

