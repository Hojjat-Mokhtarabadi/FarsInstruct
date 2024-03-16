TEMP_LIST = {
    'multiple_choice': {
        'PNLPhub/FarsTail' : [
            'can_you_infer',
            'does_this_imply',
            'do_they_relate',
            'confidence',
            'comparison',
            # 'claim_relation',
            # 'evaluate'
        ],
        'persiannlp/parsinlu_entailment' : [
            'GPT3_Style',
            'based_on_the_previous_passage',
            'can_you_infer',
            'consider_always_sometimes_never',
            'does_this_imply',
            'confidence',
            'evaluate',
            # 'claim_relation',
            'classify',
            'comparison',
            
        ],    
        'persiannlp/parsinlu_query_paraphrasing': [
            'compare_2_sents',
            'give_sent1_can_sent2_paraphrase',
            'given_sent1_sent2_paraphrase',
            'is_it_paraphrase'
        ],
        'PNLPhub/DigiMag' : [
            'classify_content',
            'does_it_belong_to_art',
            'does_it_belong_to_book',
            'does_it_belong_to_cosmetic',
            'does_it_belong_to_game',
            'does_it_belong_to_general',
            'does_it_belong_to_shop',
            'does_it_belong_to_techno',
            'what_categ_is_the_content_is'
        ],
        'persiannlp/parsinlu_sentiment': [
            'question_review_aspect',
            'question_aspect',
            'review_aspect',
            'question_aspect2'
        ],
        'pn_summary' : [
            'classify_summary',
            'classify_title',
            'select_correct_class',
            'what_category_it_belongs_to'
        ],
        'PNLPhub/digikala-sentiment-analysis' : [
            'is_avg',
            'is_bad',
            'is_good',
            'is_it_pos_or_neg',
            'is_perfect',
            'is_terrible',
            'specify_categ',
            'star_rating',
            'what_is_sentiment'
        ],
        'PNLPhub/parsinlu-multiple-choice' : [
            'category_answer',
            'category',
            'category_QA',
            'choose_the_correct_candidate',
            'which_candidate',
            'most_correct_answer'
        ],
        'PNLPhub/Persian-News' : [
            'choose_category_zs',
            'choose_label_id_and_label_zs',
            'classify_content_zs',
            'title_to_text_zs'
        ],
        'PNLPhub/snappfood-sentiment-analysis' : [
            'is_it_neg',
            'is_it_pos_or_neg',
            'is_it_pos',
            'recommendation',
            'to_which_does_belong',
            'what_is_sentiment'
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