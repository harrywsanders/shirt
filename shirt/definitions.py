model_name = 'Qwen/Qwen2.5-0.5B'
ns_target = [5,10]
ft_methods = ['shirt', 'deft', 'random']

n_aux = 200
n_epochs = 3
batch_size = 4

benchmarks = ['leaderboard_bbh_boolean_expressions',
              'leaderboard_bbh_causal_judgement',
              'leaderboard_bbh_date_understanding',
              'leaderboard_bbh_disambiguation_qa',
              'leaderboard_bbh_formal_fallacies',
              'leaderboard_bbh_geometric_shapes',
              'leaderboard_bbh_hyperbaton',
              'leaderboard_bbh_logical_deduction_five_objects',
              'leaderboard_bbh_logical_deduction_seven_objects',
              'leaderboard_bbh_logical_deduction_three_objects',
              'leaderboard_bbh_movie_recommendation',
              'leaderboard_bbh_navigate',
              'leaderboard_bbh_object_counting',
              'leaderboard_bbh_penguins_in_a_table',
              'leaderboard_bbh_reasoning_about_colored_objects',
              'leaderboard_bbh_ruin_names',
              'leaderboard_bbh_salient_translation_error_detection',
              'leaderboard_bbh_snarks',
              'leaderboard_bbh_sports_understanding',
              'leaderboard_bbh_temporal_sequences',
              'leaderboard_bbh_tracking_shuffled_objects_five_objects',
              'leaderboard_bbh_tracking_shuffled_objects_seven_objects',
              'leaderboard_bbh_tracking_shuffled_objects_three_objects',
              'leaderboard_bbh_web_of_lies']