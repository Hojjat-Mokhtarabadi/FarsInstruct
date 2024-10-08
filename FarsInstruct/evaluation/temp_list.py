TEMP_LIST = {
    'multiple_choice': {
        'PNLPhub/FarsTail' : [
            'can_you_infer',
            'does_this_imply',
            'do_they_relate',
            'confidence',
            'comparison',
            'claim_relation',
            'evaluate',
            # 'classify'
        ],
        'persiannlp/parsinlu_entailment' : [
            'GPT3_Style',
            'based_on_the_previous_passage',
            'can_you_infer',
            'does_this_imply',
            'confidence',
            'evaluate',
            'claim_relation',
            'classify',
            'comparison'
            
        ],    
        'persiannlp/parsinlu_query_paraphrasing': [
            'compare_two_sents',
            'different_or_same',
            'sucess_level',
            'relatable_or_not',
            'never_always',
            'similar_or_not',
            #'same_meaning'
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
            'choose_category',
            'choose_label_id_and_label',
            'classify_content',
            'title_to_text'
        ],
        'PNLPhub/snappfood-sentiment-analysis' : [
            'is_it_neg',
            'is_it_pos_or_neg',
            'is_it_pos',
            'recommendation',
            'to_which_does_belong',
            'what_is_sentiment',
            'possibility',
            'rate'
        ],

        "PNLPhub/C-ExaPPC": [
            "similar", 
            "different_point",
            "same_point",
        ]
  },

    'generate_until': {
        'PNLPhub/FarsTail': [
            'label_to_hypothesis',
            'label_to_premise'
        ],
        # 'persiannlp/parsinlu_entailment': [],
        # 'persiannlp/parsinlu_query_paraphrasing': [],
        'PNLPhub/DigiMag': [
            'generate_text',
            'in_which_categ_would_it_go'
        ],
        'parsinlu_reading_comprehension': [
            'give_short_answer',
            'give_short_answers',
            'question_context'
        ],
        'persiannlp/parsinlu_sentiment': [
            'aspect_category_review',
            'question_aspect_example_id',
            'question_category'
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
            'generate_comment'
        ],
        'PNLPhub/parsinlu-multiple-choice' : [
            'write_answer'
        ],
        'PNLPhub/Persian-News' : [
            'gen_category_for_content',
            'title_to_text',
            'gen_second_half'
        ],
        'PNLPhub/snappfood-sentiment-analysis' : [
            'gen_sentiment',
            'feelings'
        ],
        'SajjadAyoubi/persian_qa' : [
            'answer_Q_A',
            'answer_question_give_text',
            'generate_question',
            'generate_question_wrt_answer'
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