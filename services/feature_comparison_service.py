from typing import Dict


class FeatureComparisonService:
    @staticmethod
    def describe_diff(input_val, ref_val, metric_name, unit=""):
        if input_val is None or ref_val is None:
            return f"  - {metric_name}: Data not available\n"

        diff = input_val - ref_val
        diff_pct = (diff / ref_val * 100) if ref_val != 0 else 0

        if abs(diff_pct) < 10:  # Less than 10% difference
            return f"  - {metric_name}: Very similar ({input_val:.2f}{unit} vs {ref_val:.2f}{unit})\n"
        elif diff > 0:
            magnitude = "significantly" if abs(diff_pct) > 30 else "moderately"
            return f"  - {metric_name}: Your track is {magnitude} higher ({input_val:.2f}{unit} vs {ref_val:.2f}{unit}, +{diff_pct:.0f}%)\n"
        else:
            magnitude = "significantly" if abs(diff_pct) > 30 else "moderately"
            return f"  - {metric_name}: Your track is {magnitude} lower ({input_val:.2f}{unit} vs {ref_val:.2f}{unit}, {diff_pct:.0f}%)\n"

    @staticmethod
    def create_feature_comparison(input_features: Dict, ref_features: Dict) -> str:
        """
        Create a comparative analysis between input and reference track features
        """
        if not input_features or not ref_features:
            return "Feature comparison not available - missing feature data."

        comparison = "## Analysis Against Your Reference Track:\n\n"

        # Helper function to describe differences
        def describe_diff(input_val, ref_val, metric_name, unit=""):
            if input_val is None or ref_val is None:
                return f"  - {metric_name}: Data not available\n"

            diff = input_val - ref_val
            diff_pct = (diff / ref_val * 100) if ref_val != 0 else 0

            if abs(diff_pct) < 10:  # Less than 10% difference
                return f"  - {metric_name}: Very similar ({input_val:.2f}{unit} vs {ref_val:.2f}{unit})\n"
            elif diff > 0:
                magnitude = "significantly" if abs(diff_pct) > 30 else "moderately"
                return f"  - {metric_name}: Your track is {magnitude} higher ({input_val:.2f}{unit} vs {ref_val:.2f}{unit}, +{diff_pct:.0f}%)\n"
            else:
                magnitude = "significantly" if abs(diff_pct) > 30 else "moderately"
                return f"  - {metric_name}: Your track is {magnitude} lower ({input_val:.2f}{unit} vs {ref_val:.2f}{unit}, {diff_pct:.0f}%)\n"

        # Rhythm comparison
        rhythm_input = input_features.get("rhythm", {})
        rhythm_ref = ref_features.get("rhythm", {})

        if rhythm_input and rhythm_ref:
            comparison += "**Rhythmic Character:**\n"
            comparison += describe_diff(
                rhythm_input.get("tempo"), rhythm_ref.get("tempo"), "Tempo", " BPM"
            )
            comparison += describe_diff(
                rhythm_input.get("onset_density"),
                rhythm_ref.get("onset_density"),
                "Rhythmic Activity",
                " events/sec",
            )
            comparison += describe_diff(
                rhythm_input.get("beat_strength"),
                rhythm_ref.get("beat_strength"),
                "Beat Presence",
                "",
            )
            comparison += "\n"

        # Energy comparison
        energy_input = input_features.get("energy", {})
        energy_ref = ref_features.get("energy", {})

        if energy_input and energy_ref:
            comparison += "**Energy Profile:**\n"
            comparison += describe_diff(
                energy_input.get("dynamic_range"),
                energy_ref.get("dynamic_range"),
                "Dynamic Range",
                "",
            )
            comparison += describe_diff(
                energy_input.get("average_energy"),
                energy_ref.get("average_energy"),
                "Overall Intensity",
                "",
            )
            comparison += describe_diff(
                energy_input.get("peak_density"),
                energy_ref.get("peak_density"),
                "Energy Peaks",
                " /sec",
            )
            comparison += "\n"

        # Frequency/EQ comparison
        freq_input = input_features.get("frequency", {})
        freq_ref = ref_features.get("frequency", {})

        if freq_input and freq_ref:
            comparison += "**Frequency Distribution:**\n"
            comparison += describe_diff(
                freq_input.get("low_proportion"),
                freq_ref.get("low_proportion"),
                "Bass Content",
                "%",
            )
            comparison += describe_diff(
                freq_input.get("mid_proportion"),
                freq_ref.get("mid_proportion"),
                "Midrange Content",
                "%",
            )
            comparison += describe_diff(
                freq_input.get("high_proportion"),
                freq_ref.get("high_proportion"),
                "Treble Content",
                "%",
            )
            comparison += "\n"

        # Spectral comparison
        spectral_input = input_features.get("spectral", {})
        spectral_ref = ref_features.get("spectral", {})

        if spectral_input and spectral_ref:
            comparison += "**Tonal Character:**\n"
            comparison += describe_diff(
                spectral_input.get("avg_brightness"),
                spectral_ref.get("avg_brightness"),
                "Overall Brightness",
                " Hz",
            )
            comparison += "\n"

        comparison += "This analysis describes the measurable differences between your input track and reference track to provide context for the feedback below.\n\n"

        return comparison
