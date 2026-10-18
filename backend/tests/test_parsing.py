import xml.etree.ElementTree as ET

import pytest

from parsing import parse_query_xml

# two panels: A and B included (OR inside one panel), X excluded (invert = 1)
XML_TWO_PANELS = """
<query_definition>
  <panel>
    <invert>0</invert>
    <total_item_occurrences>1</total_item_occurrences>
    <item><item_key>A</item_key></item>
    <item><item_key>B</item_key></item>
  </panel>
  <panel>
    <invert>1</invert>
    <total_item_occurrences>1</total_item_occurrences>
    <item><item_key>X</item_key></item>
  </panel>
</query_definition>
"""

# one panel, one concept, must occur at least 3 times
XML_OCCURRENCES = """
<query_definition>
  <panel>
    <invert>0</invert>
    <total_item_occurrences>3</total_item_occurrences>
    <item><item_key>A</item_key></item>
  </panel>
</query_definition>
"""

# A appears in two panels, so it is a duplicate
XML_DUPLICATE = """
<query_definition>
  <panel>
    <invert>0</invert>
    <total_item_occurrences>1</total_item_occurrences>
    <item><item_key>A</item_key></item>
    <item><item_key>B</item_key></item>
  </panel>
  <panel>
    <invert>0</invert>
    <total_item_occurrences>1</total_item_occurrences>
    <item><item_key>A</item_key></item>
  </panel>
</query_definition>
"""

XML_DATE = """
<query_definition>
  <panel>
    <invert>0</invert>
    <total_item_occurrences>1</total_item_occurrences>
    <item>
      <item_key>A</item_key>
      <constrain_by_date>
        <date_from>2024-01-01</date_from>
        <date_to>2024-12-31</date_to>
      </constrain_by_date>
    </item>
  </panel>
</query_definition>
"""

XML_VALUE = """
<query_definition>
  <panel>
    <invert>0</invert>
    <total_item_occurrences>1</total_item_occurrences>
    <item>
      <item_key>A</item_key>
      <constrain_by_value>
        <value_operator>GT</value_operator>
        <value_constraint>5</value_constraint>
        <value_unit_of_measure>mg</value_unit_of_measure>
      </constrain_by_value>
    </item>
  </panel>
</query_definition>
"""

# date + value + occurrences (2) in the same panel: three constraints
XML_ALL_CONSTRAINTS = """
<query_definition>
  <panel>
    <invert>0</invert>
    <total_item_occurrences>2</total_item_occurrences>
    <item>
      <item_key>A</item_key>
      <constrain_by_date>
        <date_from>2024-01-01</date_from>
        <date_to>2024-12-31</date_to>
      </constrain_by_date>
      <constrain_by_value>
        <value_operator>GT</value_operator>
        <value_constraint>5</value_constraint>
        <value_unit_of_measure>mg</value_unit_of_measure>
      </constrain_by_value>
    </item>
  </panel>
</query_definition>
"""

XML_TEMPORAL = """
<query_definition>
  <panel>
    <invert>0</invert>
    <total_item_occurrences>1</total_item_occurrences>
    <item><item_key>A</item_key></item>
  </panel>
  <subquery_constraint></subquery_constraint>
  <subquery_constraint></subquery_constraint>
</query_definition>
"""


def test_included_and_excluded_concepts():
    result = parse_query_xml(XML_TWO_PANELS)

    assert result["included_concepts"] == ["A", "B"]
    assert result["excluded_concepts"] == ["X"]


def test_item_keys_are_included_then_excluded():
    result = parse_query_xml(XML_TWO_PANELS)

    assert result["item_keys"] == ["A", "B", "X"]


def test_panels_structure():
    result = parse_query_xml(XML_TWO_PANELS)
    panels = result["panels"]

    assert len(panels) == 2
    assert panels[0]["excluded"] is False
    assert panels[1]["excluded"] is True
    assert panels[0]["occurrences"] == 1
    assert [item["key"] for item in panels[0]["items"]] == ["A", "B"]
    assert [item["key"] for item in panels[1]["items"]] == ["X"]


def test_no_constraints():
    result = parse_query_xml(XML_TWO_PANELS)

    assert result["num_constraints"] == 0
    assert result["date_constraints"] == []
    assert result["temporal_query_constraints"] == 0


def test_occurrences():
    result = parse_query_xml(XML_OCCURRENCES)

    assert result["num_constraints"] == 1
    assert result["panels"][0]["occurrences"] == 3


def test_date_constraint():
    result = parse_query_xml(XML_DATE)

    assert result["date_constraints"] == [{"date_from": "2024-01-01", "date_to": "2024-12-31"}]
    assert result["num_constraints"] == 1


def test_date_constraint_in_panel_item():
    result = parse_query_xml(XML_DATE)
    item = result["panels"][0]["items"][0]

    assert item["date_from"] == "2024-01-01"
    assert item["date_to"] == "2024-12-31"


def test_value_constraint():
    result = parse_query_xml(XML_VALUE)
    item = result["panels"][0]["items"][0]

    assert result["num_constraints"] == 1
    assert item["value_operator"] == "GT"
    assert item["value_constraint"] == "5"
    assert item["value_unit"] == "mg"


def test_num_constraints_adds_all_types():
    result = parse_query_xml(XML_ALL_CONSTRAINTS)

    assert result["panels"][0]["occurrences"] == 2
    assert result["num_constraints"] == 3


def test_temporal_constraints():
    result = parse_query_xml(XML_TEMPORAL)

    assert result["temporal_query_constraints"] == 2


def test_malformed_xml_raises_parse_error():
    with pytest.raises(ET.ParseError):
        parse_query_xml("<query_definition><panel>")
