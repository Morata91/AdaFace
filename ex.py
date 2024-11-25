import os
import bcolz
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

def read_pairs(pairs_filename):
    """
    pairs.txtを読み込んでペア情報を取得する
    
    Args:
        pairs_filename: pairs.txtのパス
    Returns:
        pairs: [(name1, id1, name2, id2), ...] または [(name, id1, id2), ...]
        issame: 同一人物かどうかのリスト
    """
    pairs = []
    issame = []
    
    with open(pairs_filename, 'r') as f:
        # 最初の行はペアの数とフォールドの数
        n_pairs, n_folds = map(int, f.readline().strip().split())
        
        # 各行を読み込む
        for line in f:
            line = line.strip().split()
            if len(line) == 3:
                # 同一人物のペア
                name = line[0]
                id1 = int(line[1])
                id2 = int(line[2])
                pairs.append((name, id1, name, id2))
                issame.append(True)
            elif len(line) == 4:
                # 異なる人物のペア
                name1 = line[0]
                id1 = int(line[1])
                name2 = line[2]
                id2 = int(line[3])
                pairs.append((name1, id1, name2, id2))
                issame.append(False)
    
    return pairs, issame

def prepare_lfw_data(lfw_dir, pairs_txt, output_dir):
    """
    LFWデータをbcolz形式に変換する
    
    Args:
        lfw_dir: LFW画像が含まれるディレクトリ
        pairs_txt: pairs.txtのパス
        output_dir: 出力先のベースディレクトリ
    """
    # bcolz用の出力ディレクトリ
    bcolz_dir = os.path.join(output_dir, 'lfw')
    os.makedirs(bcolz_dir, exist_ok=True)
    os.makedirs(os.path.join(bcolz_dir, 'meta'), exist_ok=True)
    
    # ペア情報の読み込み
    print("Reading pairs.txt...")
    pairs, issame = read_pairs(pairs_txt)
    
    # 画像の読み込み
    images = []
    valid_pairs = []
    valid_issame = []
    
    print("Processing image pairs...")
    for i, (name1, id1, name2, id2) in enumerate(tqdm(pairs)):
        # 画像パスの生成
        img1_path = os.path.join(lfw_dir, name1, f"{name1}_{id1:04d}.jpg")
        img2_path = os.path.join(lfw_dir, name2, f"{name2}_{id2:04d}.jpg")
        
        try:
            img1 = process_image(img1_path)
            img2 = process_image(img2_path)
            
            # 両方の画像が正常に処理できた場合のみ追加
            images.extend([img1, img2])
            valid_pairs.append(pairs[i])
            valid_issame.append(issame[i])
            
        except Exception as e:
            print(f"Error processing pair ({img1_path}, {img2_path}): {e}")
            continue
    
    # NumPy配列に変換
    images = np.stack(images).astype(np.float32)
    issame = np.array(valid_issame)
    
    # bcolzとして保存
    bcolz.carray(images, rootdir=os.path.join(bcolz_dir), mode='w')
    
    
    # ペア情報保存
    np.save(os.path.join(bcolz_dir, 'lfw_pairs.npy'), issame)
    
    print(f"Processed {len(images)} images")
    print(f"Created {sum(issame)} positive pairs and {sum(~issame)} negative pairs")

def process_image(image_path):
    """画像の前処理"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((112, 112))  # 標準的なサイズ
    img = np.array(img).transpose(2, 0, 1)  # CHW形式に変換
    img = (img - 127.5) / 128.0  # 正規化
    return img.astype(np.float32)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--lfw_dir', type=str, default='/workspace-cloud/koki.murata/AdaFace/data/val_data/lfw_funneled')
    parser.add_argument('--pairs_txt', type=str, default='/workspace-cloud/koki.murata/AdaFace/data/val_data/lfw_funneled/pairs.txt')
    parser.add_argument('--output_dir', type=str, default='/workspace-cloud/koki.murata/AdaFace/data/val_data')
    args = parser.parse_args()
    
    prepare_lfw_data(
        lfw_dir=args.lfw_dir,
        pairs_txt=args.pairs_txt,
        output_dir=args.output_dir
    )