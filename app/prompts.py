"""The extraction instruction every model gets.

Keeping one prompt across all adapters is what makes the model comparison fair:
differences in the output reflect the model, not the prompt. The JSON shape is
enforced separately by each provider's structured-output mechanism, so the prompt
focuses on *what* to extract and the judgment calls.
"""

EXTRACTION_PROMPT = """You are assisting a federal alcohol label compliance reviewer.

Look at the label image and extract the fields below. Read the text as printed —
do not correct, expand, or normalize it. If a field is not visibly present, return null.

Fields:
- is_alcohol_label: true only if this is genuinely an alcohol beverage label. If it
  is a photo of something else (a person, an animal, a landscape, a blank page),
  return false and leave the other fields null.
- multiple_products: true if the image shows more than one DISTINCT product/label
  (e.g. several different bottles lined up). A single bottle, or a front-and-back
  pair of the same product, is NOT multiple. If true, set product_count to how many
  distinct products you see, and for each remaining field fill in the clearest, most
  readable value you can find (it may come from different products) so a human can
  review it.
- product_count: number of distinct products when multiple_products is true.
- detected_beverage_type: beer, wine, distilled spirits, seltzer, or unknown.
- brand_name
- class_type: the class/type designation (e.g. "Kentucky Straight Bourbon Whiskey").
- alcohol_content: exactly as printed (e.g. "45% Alc./Vol. (90 Proof)").
- net_contents: exactly as printed (e.g. "750 mL").
- name_address: name and address of the bottler, producer, or importer.
- country_of_origin: only if shown (imports); otherwise null.
- government_warning_text: the FULL health warning statement exactly as it appears,
  preserving capitalization and wording. Null if there is no warning.
- government_warning_bold_visual: your visual judgment of whether the words
  "GOVERNMENT WARNING" appear bolder/heavier than the surrounding warning text.
  This is a best-effort visual impression, not a measurement. Null if no warning.
- notes: briefly flag anything that affected your reading (glare, angle, crop,
  partial text), or null.

If the image is rotated or photographed at an angle, still do your best to read it.
"""