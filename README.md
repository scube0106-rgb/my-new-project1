<!-- This is the markdown template for the final project of the Building AI course, 
created by Reaktor Innovations and University of Helsinki. 
Copy the template, paste it to your GitHub README and edit! -->

# AI-Powered Career Path Recommender
Final project for the Building AI course

## Summary
Build an AI system that analyzes a user's skills, interests, and preferences to recommend personalized career paths or job roles. 
This project aims to help users—especially students and professionals—identify suitable career paths based on their skills, interests, and preferences using AI.

## Background
In today’s fast-paced and evolving job market, individuals—especially students and career changers—face the challenge of choosing a career that aligns with their skills, interests, and aspirations. 
Traditional career counseling methods are limited by accessibility, scalability, and personalization.
AI offers an opportunity to transform career guidance by analyzing individual profiles and massive labor market data to suggest tailored career paths. 
An AI-powered career recommender system can bridge the gap between personal potential and market opportunities using data-driven insights.

## How is it used?
The system acts as a virtual career counselor, enabling users to:
-Input their skills, interests, and preferences.
-Receive career path suggestions aligned with their profile.
-Explore reasons behind recommendations (e.g., skill matches, growth trends).
-Optionally access learning resources or job listings.

This kind of tool can be used by:
-Students choosing academic or vocational directions.
-Professionals considering a career pivot.
-Educational institutions and career services.

This is how you create code examples:
    ```
    pip install scikit-learn pandas numpy
    import pandas as pd
    {
    # Sample career data
    careers = pd.DataFrame({
        'career': ['Data Scientist', 'Web Developer', 'Digital Marketer', 'UX Designer'],
        'skills': [
            'python machine learning statistics data visualization',
            'html css javascript react frontend backend',
            'seo content marketing analytics branding',
            'user research wireframing prototyping usability design thinking'
        ]
    })

    # Simulated user input
    user_input = "I enjoy working with data, coding in Python, and analyzing trends"
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Combine all text (career skills + user input)
    all_texts = careers['skills'].tolist() + [user_input]

    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # Compare user input (last vector) to all career skill vectors
    user_vector = tfidf_matrix[-1]
    career_vectors = tfidf_matrix[:-1]

    # Compute cosine similarities
    similarities = cosine_similarity(user_vector, career_vectors).flatten()

    # Attach scores to the dataframe
    careers['match_score'] = similarities
    
    # Sort by similarity score
    recommendations = careers.sort_values(by='match_score', ascending=False)
    
    # Show top 3 recommendations
    print("Top Career Recommendations:")
    print(recommendations[['career', 'match_score']].head(3))
    }

    Output (Example)
    ------------------------------------------
    | Top Career Recommendations:|           |
    ------------------------------------------ 
    |   Career                   |Match_score|
    ------------------------------------------
    |0  Data Scientist           |0.512      |
    |1  Digital Marketer         |0.212      |
    |2  Web Developer            |0.143      |
    ------------------------------------------

## Data sources and AI methods
To train and support the system, we can use the following data:
-O*NET Dataset (U.S. Department of Labor)
Contains job roles, required skills, tasks, and knowledge areas.
Publicly available and widely used for career research.
-Kaggle Datasets (e.g., career data, job titles, skill matrices)
Often includes labeled job descriptions and career transitions.
-Online Job Listings APIs
E.g., LinkedIn, Indeed, or Glassdoor for real-time market trends.
-Self-generated user input
Used for testing and refining the recommendation engine.

The system leverages the following techniques:
    ___________________________________________________________________________________________________________________________________
    |Component                    |Method	                                          |Description                                    |
    |_________________________________________________________________________________________________________________________________|
    |Text Preprocessing	          |NLP (Tokenization, Stopword removal)	              |Clean and prepare free-text user input         |
    
    |Skill/Interest Representatio |Embedding (e.g., TF-IDF, BERT)	                  |Convert skills and interests to vectors        |
    |
    |Matching & Recommendations	  |Similarity Matching (Cosine Similarity, KNN)       |Identify the best-matching career paths        |
    |                             |or Classifier (Random Forest, Logistic Regression) |                                               |
  
    |Optional: Explanation        |Attention or Feature Importance	                  |Help users understand why career is recommended|
    |_________________________________________________________________________________________________________________________________|

## Challenges
1.Data Quality & Consistency:
Job data varies in format and terminology; mapping user inputs to standardized job skills is tricky.
2.Subjectivity of Preferences:
Interests and preferences are often qualitative and nuanced, making them hard to quantify.
3.Cold Start Problem:
For new users or rare skill combinations, recommendations may be less accurate.
4.Bias and Fairness:
Bias in historical job data can lead to biased recommendations (e.g., by gender or region).
5.Scalability:
Handling large, real-time datasets requires performance optimization.

## What next?
1.Personalization at Scale:
Use reinforcement learning to adapt recommendations based on user feedback.
2.Resume & Document Parsing:
Allow users to upload resumes for automatic skill extraction.
3.Live Job Integration:
Link recommendations to current job openings with matching scores.
4.Localization & Diversity:
Expand to include regional job markets and culturally relevant career options.
5.AI Explainability:
Provide transparent explanations for every recommendation to build trust.

## Acknowledgments
We would like to acknowledge:
-O*NET for providing comprehensive occupational data.
-Kaggle Community for accessible datasets and career-related challenges.
-OpenAI and Hugging Face for pre-trained language models and embeddings.
-Instructors/Mentors of the Building AI course for guidance and inspiration.
