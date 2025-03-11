import os
from flask import Flask, render_template, request, flash
import numpy as np
import requests
import markdown

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a secure key

##############################################################################
# 1) Configure DeepSeek API Settings
##############################################################################
DEEPSEEK_API_KEY = "sk-9ed3cb54e8804ec29c461161a0e30b95"  # Replace with your actual DeepSeek API key
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"  # Replace with the actual DeepSeek endpoint

##############################################################################
# 2) Recommended Ranges for Design Parameters (Illustrative)
##############################################################################
# Format: parameter: (min, max, unit)
recommended_ranges = {
    "H": (10.0, 26.0, "ft"),
    "B": (3.0, 10.0, "ft"),
    "Lt": (1.5, 5.0, "ft"),
    "Lh": (1.5, 6.5, "ft"),
    "fc": (3000.0, 7000.0, "psi"),
    "fy": (60000.0, 90000.0, "psi"),
    "gamma_c": (145.0, 150.0, "pcf"),
    "gamma_s": (110.0, 120.0, "pcf"),
    "Ka": (0.3, 0.5, ""),
    "Kp": (2.0, 4.0, ""),
    "q_all": (2000.0, 6000.0, "psf"),
    "q": (100.0, 400.0, "psf")
}


##############################################################################
# 3) Helper: Calculate Design Results
##############################################################################
def perform_calculations(data):
    H = float(data.get("H"))
    B = float(data.get("B"))
    Lt = float(data.get("Lt"))
    Lh = float(data.get("Lh"))
    fc = float(data.get("fc"))
    fy = float(data.get("fy"))
    gamma_c = float(data.get("gamma_c"))
    gamma_s = float(data.get("gamma_s"))
    Ka = float(data.get("Ka"))
    Kp = float(data.get("Kp"))
    q_all = float(data.get("q_all"))
    q = float(data.get("q"))

    # Overturning
    OverturningMoment = 0.5 * Ka * gamma_s * H ** 3
    ResistingMoment = gamma_c * B * H ** 2 / 2
    FOS_Overturning = ResistingMoment / OverturningMoment if OverturningMoment != 0 else float('inf')

    # Sliding
    SlidingForce = Ka * gamma_s * H ** 2
    ResistingForce = Kp * B * H
    FOS_Sliding = ResistingForce / SlidingForce if SlidingForce != 0 else float('inf')

    # Bearing
    BaseReaction = (gamma_c * B * H) + (q * B)
    BearingPressure = BaseReaction / B if B != 0 else float('inf')
    FOS_Bearing = q_all / BearingPressure if BearingPressure != 0 else float('inf')

    # Simplified ACI Check
    d = H - 0.05
    Mu = OverturningMoment / 1.5
    Mn = 0.9 * fc * 0.85 * d ** 2
    ACI_Check = "Structurally Safe" if Mn > Mu else "Redesign Required"

    return {
        "FOS_Overturning": f"{FOS_Overturning:.2f}",
        "FOS_Sliding": f"{FOS_Sliding:.2f}",
        "FOS_Bearing": f"{FOS_Bearing:.2f}",
        "ACI_Check": ACI_Check
    }


##############################################################################
# 4) Helper: Get AI Feedback and Convert Markdown to HTML
##############################################################################
def get_ai_feedback(data, calc_results):
    # Check inputs against recommended ranges
    input_review = []
    for param, (min_val, max_val, unit) in recommended_ranges.items():
        val = float(data.get(param))
        if not (min_val <= val <= max_val):
            input_review.append(
                f"- {param} = {val:.2f} {unit} is outside the typical range [{min_val}-{max_val} {unit}]."
            )
    if not input_review:
        range_check_summary = "All inputs are within typical recommended ranges.\n"
    else:
        range_check_summary = "Some inputs are outside typical recommended ranges:\n" + "\n".join(input_review) + "\n"

    # Compose prompt
    prompt = (
        "I have a retaining wall design with the following parameters:\n"
        f"Stem Height (H): {data.get('H')} ft\n"
        f"Base Width (B): {data.get('B')} ft\n"
        f"Toe Length (Lt): {data.get('Lt')} ft\n"
        f"Heel Length (Lh): {data.get('Lh')} ft\n"
        f"Concrete Strength (fc): {data.get('fc')} psi\n"
        f"Steel Yield Strength (fy): {data.get('fy')} psi\n"
        f"Concrete Unit Weight (gamma_c): {data.get('gamma_c')} pcf\n"
        f"Soil Unit Weight (gamma_s): {data.get('gamma_s')} pcf\n"
        f"Active Pressure Coefficient (Ka): {data.get('Ka')}\n"
        f"Passive Pressure Coefficient (Kp): {data.get('Kp')}\n"
        f"Allowable Soil Bearing Capacity (q_all): {data.get('q_all')} psf\n"
        f"Surcharge Load (q): {data.get('q')} psf\n\n"
        "Calculated Results:\n"
        f" - FOS Overturning: {calc_results['FOS_Overturning']}\n"
        f" - FOS Sliding: {calc_results['FOS_Sliding']}\n"
        f" - FOS Bearing: {calc_results['FOS_Bearing']}\n"
        f" - ACI Check: {calc_results['ACI_Check']}\n\n"
        "Typical Range Check:\n"
        f"{range_check_summary}\n"
        "Please analyze these parameters and results and provide detailed feedback on how to optimize the design for safety, cost efficiency, and performance. "
        "Include suggestions to improve the design if necessary, especially where inputs are outside the recommended range."
    )

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",  # Replace with correct model name per DeepSeek docs
        "messages": [{"role": "user", "content": prompt}]
    }
    response_api = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
    response_json = response_api.json()
    if "choices" in response_json and len(response_json["choices"]) > 0:
        raw_text = response_json["choices"][0]["message"]["content"]
        # Convert markdown to HTML for nice formatting
        ai_feedback_html = markdown.markdown(raw_text)
        return ai_feedback_html
    else:
        return "<p>Unexpected API response format.</p>"


##############################################################################
# 5) Main Route
##############################################################################
@app.route("/", methods=["GET", "POST"])
def index():
    calc_results = None
    ai_feedback_html = None
    form_data = {}
    if request.method == "POST":
        # Retrieve form data
        keys = ["H", "B", "Lt", "Lh", "fc", "fy", "gamma_c", "gamma_s", "Ka", "Kp", "q_all", "q"]
        try:
            for key in keys:
                form_data[key] = request.form.get(key)
                float(form_data[key])  # validate conversion
        except (ValueError, TypeError):
            flash("Please enter valid numerical values.")
            return render_template("index.html", form_data=form_data)

        calc_results = perform_calculations(form_data)
        action = request.form.get("action")
        if action == "ai_feedback":
            ai_feedback_html = get_ai_feedback(form_data, calc_results)
    return render_template("index.html", form_data=form_data, calc_results=calc_results, ai_feedback=ai_feedback_html)


if __name__ == "__main__":
   import os
port = int(os.environ.get("PORT", 5000))  # Railway provides a port dynamically
app.run(host="0.0.0.0", port=port)

