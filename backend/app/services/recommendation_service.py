from app.schemas.risk import PatientInput


def generate_recommendations(data: PatientInput, probability: float) -> list[str]:
    recommendations: list[str] = []

    high_danger_odors = {"c", "f", "m", "p", "s", "y"}
    if data.odor in high_danger_odors:
        recommendations.append(
            "The selected odor is commonly linked to poisonous mushrooms. Treat this sample as unsafe."
        )

    if data.spore_print_color == "r":
        recommendations.append(
            "Green spore print is a strong warning signal. Do not consume this mushroom."
        )

    if data.gill_size == "n":
        recommendations.append(
            "Narrow gills increase toxicity risk in this dataset. Use expert verification before any handling."
        )

    if data.ring_type in {"e", "n"}:
        recommendations.append(
            "This ring pattern appears more often in poisonous classes. Handle with caution."
        )

    if data.habitat in {"u", "w"}:
        recommendations.append(
            "Urban and waste habitats can increase uncertainty. Avoid consumption when confidence is not very high."
        )

    if probability >= 0.7:
        recommendations.append(
            "Model confidence is high for toxicity risk. Mark as potentially poisonous and avoid ingestion."
        )
    elif probability <= 0.3:
        recommendations.append(
            "Model indicates lower toxicity risk, but never eat wild mushrooms without professional confirmation."
        )
    else:
        recommendations.append(
            "This is an uncertain zone. Request a manual expert check before making safety decisions."
        )

    return recommendations
