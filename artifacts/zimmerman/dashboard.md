# Zimmerman Toolkit Analysis -- LEMURS Semester Simulator

Generated: 2026-02-21 04:42:13

## Sobol Global Sensitivity
- Base samples: 32
- Total sims: 448
- Parameters: ['nature_rx', 'exercise_rx', 'therapy_rx', 'sleep_hygiene', 'caffeine_reduction', 'academic_load', 'age', 'gender', 'emotional_stability', 'trauma_load', 'mh_diagnosis', 'baseline_chronotype']
- **sleep_quality_tst_mean** top S1: sleep_hygiene=44.525, caffeine_reduction=-4.297, nature_rx=0.028
- **sleep_quality_tst_final** top S1: sleep_hygiene=26.388, caffeine_reduction=-2.446, nature_rx=0.036
- **sleep_quality_tst_min** top S1: sleep_hygiene=29.878, caffeine_reduction=-1.732, nature_rx=0.016
- **sleep_quality_sleep_quality_mean** top S1: sleep_hygiene=152.415, caffeine_reduction=-15.304, nature_rx=0.059
- **sleep_quality_sleep_quality_final** top S1: sleep_hygiene=60.069, caffeine_reduction=-5.882, nature_rx=0.043

## Falsification
- Tests run: 200
- Violations: 0
- Violation rate: 0.0%

## Contrastive Analysis
- Flip pairs found: 0

## POSIWID Alignment
- Mean overall alignment: 0.727
- Direction accuracy: 0.900
- Magnitude accuracy: 0.554

## PDS Mapping
- sleep_quality_tst_mean variance explained: 1.000
- sleep_quality_tst_final variance explained: 1.000
- sleep_quality_tst_min variance explained: 1.000
- sleep_quality_sleep_quality_mean variance explained: 1.000
- sleep_quality_sleep_quality_final variance explained: 1.000
- sleep_quality_sleep_debt_cumulative variance explained: 1.000
- sleep_quality_social_jetlag_mean variance explained: 1.000
- sleep_quality_shape_cluster1_fraction variance explained: 0.907
- stress_anxiety_pss_mean variance explained: 1.000
- stress_anxiety_pss_final variance explained: 1.000
- stress_anxiety_pss_slope variance explained: 1.000
- stress_anxiety_pss_peak variance explained: 0.934
- stress_anxiety_pss_time_above_threshold variance explained: 1.000
- stress_anxiety_gad7_mean variance explained: 0.858
- stress_anxiety_gad7_peak variance explained: 0.725
- stress_anxiety_gad7_days_above_10 variance explained: 0.660
- stress_anxiety_anxiety_transitions_count variance explained: 0.548
- stress_anxiety_depression_mean variance explained: 1.000
- stress_anxiety_depression_final variance explained: 1.000
- physiological_rhr_mean variance explained: 1.000
- physiological_rhr_slope variance explained: 1.000
- physiological_hrv_mean variance explained: 1.000
- physiological_hrv_final variance explained: 1.000
- physiological_hrv_slope variance explained: 1.000
- physiological_arr_mean variance explained: 1.000
- physiological_arr_slope variance explained: 1.000
- physiological_dac_min variance explained: 1.000
- intervention_response_pss_benefit variance explained: 1.000
- intervention_response_hrv_benefit variance explained: 1.000
- intervention_response_wellbeing_gain variance explained: 1.000
- intervention_response_nature_dose_response variance explained: 0.993
- intervention_response_cost_effectiveness variance explained: 1.000

## Locality Profile
- Simulations: 190
- Effective horizon: 0.8

## Dashboard Summary
- Coverage: 0/0 tools (0%)
### Recommendations
- Most influential parameters: sleep_hygiene, nature_rx, therapy_rx. Focus LLM prompts on these for maximum impact.
- Interaction strength is 5.211 -- parameters interact non-additively. Consider joint parameter prompts rather than independent per-parameter generation.
- Worst-aligned output keys: intervention_response_pss_benefit, sleep_quality_social_jetlag_mean, stress_anxiety_gad7_mean. These diverge most from intended outcomes.
- Most causal parameters: most_causal_params, most_connected_outputs.
