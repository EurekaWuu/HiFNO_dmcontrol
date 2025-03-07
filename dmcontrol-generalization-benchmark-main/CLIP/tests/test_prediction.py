import torch
import clip
from PIL import Image

def predict_image(image_path, candidate_descriptions):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)

    text_inputs = clip.tokenize(candidate_descriptions).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)


    results = [(desc, conf.item()) for desc, conf in zip(candidate_descriptions, similarity[0])]

    results.sort(key=lambda x: x[1], reverse=True)
    
    return results


if __name__ == "__main__":
    image_path = "/mnt/lustre/GPU4/home/wuhanpeng/RL-ViGen/CLIP/tests/2.jpg"
    
    descriptions = [
    "The mannequin’s torso is upright with a slight forward lean, while one leg is extended behind for maximum push-off, and the other leg is lifting towards the front for the next step.",
    "In this stride, the mannequin's torso remains straight and stable, with one leg bent at the knee in mid-air while the other leg drives forward, ensuring efficient movement and balance.",
    "With its torso aligned, the mannequin is lifting one leg in front of the body while the other leg extends backward, demonstrating an efficient running posture with proper leg extension and cadence.",
    "The mannequin maintains an upright posture, with its knees bent and one leg pushing off the ground while the other leg moves toward the front, ensuring smooth, rhythmic strides.",
    "The torso is held firm and steady, while one leg is fully extended behind, and the other leg is bent, bringing the knee high to prepare for the next stride in a fluid running motion.",
    "As the mannequin runs, its torso stays balanced and straight, with alternating leg movements—one leg propels the body forward while the other leg swings back to maximize speed and efficiency."
    ]
    
    results = predict_image(image_path, descriptions)
    
    print("\n预测结果及置信度:")
    print("-" * 40)
    for description, confidence in results:
        print(f"{description:<20}: {confidence:.2f}%")



'''

CUDA_VISIBLE_DEVICES=6 python /mnt/lustre/GPU4/home/wuhanpeng/RL-ViGen/CLIP/tests/test_prediction.py

'''