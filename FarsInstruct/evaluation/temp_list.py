TEMP_LIST = {
    'multiple_choice': {
        'PNLPhub/FarsTail' : [
            'can_you_infer_zs',
            'does_this_imply_zs',
            'do_they_relate_fs'
        ],
        'persiannlp/parsinlu_entailment' : [
            'GPT3_Style_fs',
            'based_on_the_previous_passage_zs',
            'can_you_infer_zs',
            'consider_always_sometimes_never_fs',
            'does_this_imply_fs'
        ],
        'PNLPhub/DigiMag' : [
            'classify_content_fs',
            'does_it_belong_to_art_zs',
            'does_it_belong_to_book_zs',
            'does_it_belong_to_cosmetic_zs',
            'does_it_belong_to_game_zs',
            'does_it_belong_to_general_zs',
            'does_it_belong_to_shop_zs',
            'does_it_belong_to_techno_zs',
            'what_categ_is_the_content_is_zs'
        ],
        'persiannlp/parsinlu_sentiment': [
            'question_aspect_fs',
            'review_aspect_zs'
        ],
        'pn_summary' : [
            'classify_summary_fs',
            'classify_title_fs',
            'select_correct_class_zs',
            'what_category_it_belongs_to_zs'
        ],
        'PNLPhub/digikala-sentiment-analysis' : [
            'is_avg_fs',
            'is_bad_fs',
            'is_good_fs',
            'is_it_pos_or_neg_fs',
            'is_perfect_fs',
            'is_terrible_fs',
            'specify_categ_zs',
            'star_rating_fs',
            'what_is_sentiment_fs'
        ],
        'PNLPhub/parsinlu-multiple-choice' : [
            'category_answer_fs',
            'category_zs',
            'choose_the_correct_candidate_fs',
            'which_candidate_zs'
        ],
        'PNLPhub/Persian-News' : [
            'choose_category_zs',
            'choose_label_id_and_label_zs',
            'classify_content_zs',
            'title_to_text_zs'
        ],
        'PNLPhub/snappfood-sentiment-analysis' : [
            'is_it_neg_zs',
            'is_it_pos_or_neg_fs',
            'is_it_pos_zs',
            'recommendation_fs',
            'to_which_does_belong_zs',
            'what_is_sentiment_fs'
        ]
  },








    'generate_until': {
        'PNLPhub/FarsTail': [
            'label_to_hypothesis_zs',
            'label_to_premise_zs'
        ],
        'PNLPhub/DigiMag': [
            'generate_class_fs',
            'in_which_categ_would_it_go_fs'
        ],
        'parsinlu_reading_comprehension': [
            'give_short_answer_fs',
            'give_short_answers_zs',
            'question_context_zs'
        ],
        'persiannlp/parsinlu_query_paraphrasing': [
            'compare_2_sents_fs',
            'gen_new_paraphrase_fs',
            'give_sent1_can_sent2_paraphrase_zs',
            'given_sent1_sent2_paraphrase_fs',
            'is_it_paraphrase_zs'
        ],
        'persiannlp/parsinlu_sentiment': [
            'aspect_category_review_zs',
            'question_aspect_example_id_fs',
            'question_category_zs'
        ],
        'pn_summary' : [
            'gen_article_with_summary_zs',
            'gen_sent_with_title_fs',
            'gen_sum_with_category_article_zs',
            'gen_sum_with_title_article_zs',
            'gen_title_with_summary_fs',
            'given_article_summarize_zs',
            'summarize_the_article_zs'
        ],
        'PNLPhub/digikala-sentiment-analysis' : [
            'generate_sentiment_fs'
        ],
        'PNLPhub/parsinlu-multiple-choice' : [
            'write_answer_fs'
        ],
        'PNLPhub/Persian-News' : [
            'gen_category_for_content_zs'
        ],
        'PNLPhub/snappfood-sentiment-analysis' : [
            'gen_sentiment_fs'
        ],
        'SajjadAyoubi/persian_qa' : [
            'answer_Q_A_zs',
            'answer_question_give_text_fs',
            'generate_question_fs',
            'generate_question_wrt_answer_fs'
        ],
        'SLPL/syntran-fa' : [
            'answer_question_fs',
            'gen_fluent_with_short_fs',
            'gen_q_with_long_short_ans_zs',
            'gen_short_ans_with_long_fs',
            'gen_short_long_ans_zs'
        ],
        'wiki_summary' : [
            'summarize_article_zs',
            'title_summary_zs',
            'write_article_summary_fs',
            'write_article_zs',
            'write_title_zs'
        ]
    }
}