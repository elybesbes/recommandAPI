import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import os
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import GlobalMaxPooling2D
from sklearn.metrics.pairwise import cosine_similarity
import keras
from sklearn.preprocessing import OneHotEncoder


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


img_width = 224
img_height = 224
chnls = 3
vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(img_width, img_height, chnls))
vgg16.trainable=False
vgg16_model = keras.Sequential([vgg16, GlobalMaxPooling2D()])


embedding_features_df = pd.read_csv('./Clothes_DataSet/EmbeddingFeatures.csv')
clothes_features_df = pd.read_csv('./Clothes_DataSet/ClothesFeatures.csv')
image_folder_path = './Clothes_DataSet/images'

def one_hot_encode_features(df):
    enc = OneHotEncoder()
    encoded_features = enc.fit_transform(df[['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'activity']])
    return pd.DataFrame(encoded_features.toarray())

encoded_features_df = one_hot_encode_features(clothes_features_df)

def predict(model, img_name):
    img = image.load_img(os.path.join(image_folder_path, img_name), target_size=(img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return model.predict(img)

def transform_img(img_name):
    try:
        sample_image = predict(vgg16_model, img_name)
        sample_image_flattened = sample_image.flatten()
        row_idx = clothes_features_df.index[clothes_features_df['image'] == img_name]
        if row_idx.empty:
            raise ValueError(f"Image {img_name} not found in dataset.")
        row_idx = row_idx[0]
        sample_encoded_features = encoded_features_df.iloc[row_idx].to_numpy()
        sample_image_with_features = np.hstack([sample_image_flattened, sample_encoded_features])
        return sample_image_with_features
    except Exception as e:
        raise RuntimeError(f"Error transforming image {img_name}: {e}")

def normalize_sim(similarity):
    """Normalize similarity results."""
    x_min = similarity.min(axis=1)
    x_max = similarity.max(axis=1)
    norm = (similarity - x_min) / (x_max - x_min)[:, np.newaxis]
    return norm

def get_recommendations(df, similarity, input_image_name, gender_of_input_image):
    """ Return the top 5 most similar products, excluding the input image, and matching the gender """
    gender_of_input_image = clothes_features_df.loc[clothes_features_df['image'] == input_image_name, 'gender'].values[0]
    subcat_of_input_image = clothes_features_df.loc[clothes_features_df['image'] == input_image_name, 'subCategory'].values[0]
    sim_scores = list(enumerate(similarity[0]))
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = [score for score in sim_scores if score[1] < 1 
                  and df['image'].iloc[score[0]] != input_image_name 
                  and df['gender'].iloc[score[0]] == gender_of_input_image 
                  and df['subCategory'].iloc[score[0]] == subcat_of_input_image]
    
    sim_scores = sim_scores[:3]
    
    cloth_indices = [i[0] for i in sim_scores]

    return df['image'].iloc[cloth_indices]


class RecommendationRequest(BaseModel):
    input_image_name: str

@app.post("/recommendSameType")
def get_recommendations_api(request: RecommendationRequest):
    input_image_name = request.input_image_name

    if input_image_name not in clothes_features_df['image'].values:
        raise HTTPException(status_code=404, detail=f"Image {input_image_name} not found in the dataset.")

    try:
        sample_image_with_features = transform_img(input_image_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the image: {str(e)}")

    sample_similarity = cosine_similarity([sample_image_with_features], embedding_features_df)
    sample_similarity_norm = normalize_sim(sample_similarity)

    gender_of_input_image = clothes_features_df.loc[
        clothes_features_df['image'] == input_image_name, 'gender'
    ].values[0]

    try:
        recommendations = get_recommendations(
            clothes_features_df, 
            sample_similarity_norm, 
            input_image_name, 
            gender_of_input_image
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching recommendations: {str(e)}")

    recommendation_list = recommendations.to_list()

    return {"recommendations": recommendation_list}





def get_outfit_recommendations(df, df_one_hot, input_image_name):
    input_item = df[df['image'] == input_image_name]
    
    if input_item.empty:
        raise ValueError(f"Input image '{input_image_name}' not found in the dataset.")
    
    input_item = input_item.iloc[0]
    input_subcategory = input_item['subCategory']
    input_gender = input_item['gender']
    
    if input_subcategory not in df_one_hot.index:
        raise ValueError(f"Subcategory '{input_subcategory}' not found in the one-hot encoded DataFrame.")

    # Get categories from matching_df that have 1 for the input_subcategory
    matching_categories = df_one_hot[df_one_hot[input_subcategory] == 1].index.tolist()
    
    recommendations = {}
    
    for category in matching_categories:
        category_candidates = df[(df['gender'] == input_gender) & (df['subCategory'] == category)]
        
        if not category_candidates.empty:
            recommendations[category] = category_candidates.sample(1)  # Sample one random item
    
    return recommendations





matching_df = pd.read_csv('./Clothes_DataSet/MatchingCategories.csv')
matching_df.set_index('Input', inplace=True)

@app.post("/recommendOutfit")
def get_outfit(request: RecommendationRequest):
    input_image_name = request.input_image_name

    if input_image_name not in clothes_features_df['image'].values:
        raise HTTPException(status_code=404, detail=f"Image {input_image_name} not found in the dataset.")
    
    try:
        recommendations = get_outfit_recommendations(
            clothes_features_df,
            matching_df,
            input_image_name
        )
    except asyncio.CancelledError:
        raise HTTPException(status_code=408, detail="Request timed out or was canceled.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching recommendations: {str(e)}")
    
    print(recommendations)
    recommendation_list = {category: item['image'].values[0] for category, item in recommendations.items()}

    return {"recommendations": recommendation_list}