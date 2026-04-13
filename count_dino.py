import os
import json
from pycocotools.coco import COCO
from pathlib import Path

def count_dataset_stats():
    """统计训练和验证数据集的图片和标注数量"""
    
    # 使用与 train_detector.py 相同的配置
    class Config:
        REPO_DIR = "."
        # 单疾病训练配置（与 train_detector.py 一致）
        IMAGE_DIR = "../Dataset/Caries/image"
        TRAIN_JSON = "coco/Caries/train.json"
        VAL_JSON = "coco/Caries/val.json"
        SINGLE_CAT_ID = 1
        DROP_EMPTY = True
    
    def build_category_map(train_json, single_cat_id=None):
        """与 train_detector.py 相同的 category_map 构建函数"""
        coco = COCO(train_json)
        ann_ids = coco.getAnnIds()
        anns = coco.loadAnns(ann_ids) if ann_ids else []
        used_cat_ids = sorted({a['category_id'] for a in anns})
        cat_ids = used_cat_ids if used_cat_ids else coco.getCatIds()
        if single_cat_id is not None:
            if single_cat_id not in cat_ids:
                print(f"Warning: single_cat_id {single_cat_id} not found in {train_json}. Available cat ids: {cat_ids}")
            return {single_cat_id: 1}
        cat_ids_sorted = sorted(cat_ids)
        category_map = {old: i + 1 for i, old in enumerate(cat_ids_sorted)}
        return category_map
    
    def count_coco_dataset(img_folder, ann_file, category_map=None, drop_empty=False):
        """统计单个 COCO 数据集的信息"""
        print(f"\n{'='*60}")
        print(f"统计数据集: {ann_file}")
        print(f"图片目录: {img_folder}")
        print(f"{'='*60}")
        
        # 加载 COCO 标注
        coco = COCO(ann_file)
        
        # 原始统计
        all_img_ids = list(sorted(coco.imgs.keys()))
        all_ann_ids = coco.getAnnIds()
        all_anns = coco.loadAnns(all_ann_ids) if all_ann_ids else []
        
        print(f"原始数据统计:")
        print(f"  图片总数: {len(all_img_ids)}")
        print(f"  标注总数: {len(all_anns)}")
        print(f"  类别数量: {len(coco.cats)}")
        
        # 按类别统计标注
        print(f"\n按类别统计原始标注:")
        for cat_id, cat_info in coco.cats.items():
            cat_anns = [a for a in all_anns if a['category_id'] == cat_id]
            print(f"  类别 {cat_id} ({cat_info['name']}): {len(cat_anns)} 个标注")
        
        # 应用 category_map 过滤
        if category_map:
            filtered_anns = [a for a in all_anns if a['category_id'] in category_map]
            print(f"\n应用 category_map 过滤后:")
            print(f"  保留的标注数: {len(filtered_anns)}")
            print(f"  过滤掉的标注数: {len(all_anns) - len(filtered_anns)}")
            print(f"  category_map: {category_map}")
        else:
            filtered_anns = all_anns
        
        # 应用 drop_empty 过滤
        if drop_empty:
            # 统计每张图片的标注数
            img_ann_counts = {}
            for ann in filtered_anns:
                img_id = ann['image_id']
                img_ann_counts[img_id] = img_ann_counts.get(img_id, 0) + 1
            
            # 有标注的图片
            images_with_anns = [img_id for img_id in all_img_ids if img_ann_counts.get(img_id, 0) > 0]
            images_without_anns = [img_id for img_id in all_img_ids if img_id not in images_with_anns]
            
            print(f"\n应用 drop_empty={drop_empty} 过滤后:")
            print(f"  有标注的图片数: {len(images_with_anns)}")
            print(f"  无标注的图片数: {len(images_without_anns)}")
            print(f"  最终使用的图片数: {len(images_with_anns)}")
            
            # 检查图片文件是否存在
            existing_images = 0
            missing_images = []
            for img_id in images_with_anns:
                img_info = coco.loadImgs(img_id)[0]
                file_name = img_info['file_name']
                img_path = os.path.join(img_folder, file_name)
                
                # 兼容多疾病文件夹结构
                if not os.path.exists(img_path):
                    found = False
                    for root, dirs, files in os.walk(img_folder):
                        if file_name in files:
                            found = True
                            break
                    if found:
                        existing_images += 1
                    else:
                        missing_images.append(file_name)
                else:
                    existing_images += 1
            
            print(f"\n图片文件检查:")
            print(f"  存在的图片文件: {existing_images}")
            print(f"  缺失的图片文件: {len(missing_images)}")
            if missing_images:
                print(f"  缺失的文件示例: {missing_images[:5]}")
                if len(missing_images) > 5:
                    print(f"  ... 共 {len(missing_images)} 个缺失文件")
            
            return {
                'total_images': len(all_img_ids),
                'total_annotations': len(all_anns),
                'filtered_annotations': len(filtered_anns),
                'images_with_annotations': len(images_with_anns),
                'images_without_annotations': len(images_without_anns),
                'existing_image_files': existing_images,
                'missing_image_files': len(missing_images),
                'category_map': category_map
            }
        else:
            # 不丢弃空标注图片
            print(f"\n应用 drop_empty={drop_empty} (保留所有图片)")
            print(f"  最终使用的图片数: {len(all_img_ids)}")
            
            # 检查图片文件是否存在
            existing_images = 0
            missing_images = []
            for img_id in all_img_ids:
                img_info = coco.loadImgs(img_id)[0]
                file_name = img_info['file_name']
                img_path = os.path.join(img_folder, file_name)
                
                if not os.path.exists(img_path):
                    found = False
                    for root, dirs, files in os.walk(img_folder):
                        if file_name in files:
                            found = True
                            break
                    if found:
                        existing_images += 1
                    else:
                        missing_images.append(file_name)
                else:
                    existing_images += 1
            
            print(f"\n图片文件检查:")
            print(f"  存在的图片文件: {existing_images}")
            print(f"  缺失的图片文件: {len(missing_images)}")
            if missing_images:
                print(f"  缺失的文件示例: {missing_images[:5]}")
            
            return {
                'total_images': len(all_img_ids),
                'total_annotations': len(all_anns),
                'filtered_annotations': len(filtered_anns),
                'images_with_annotations': len([img_id for img_id in all_img_ids 
                                               if any(a['image_id'] == img_id for a in filtered_anns)]),
                'images_without_annotations': len([img_id for img_id in all_img_ids 
                                                  if not any(a['image_id'] == img_id for a in filtered_anns)]),
                'existing_image_files': existing_images,
                'missing_image_files': len(missing_images),
                'category_map': category_map
            }
    
    # 构建 category_map（与 train_detector.py 相同）
    category_map = build_category_map(Config.TRAIN_JSON, single_cat_id=Config.SINGLE_CAT_ID)
    
    # 统计训练集
    train_stats = count_coco_dataset(
        Config.IMAGE_DIR, 
        Config.TRAIN_JSON, 
        category_map=category_map, 
        drop_empty=Config.DROP_EMPTY
    )
    
    # 统计验证集
    val_stats = count_coco_dataset(
        Config.IMAGE_DIR, 
        Config.VAL_JSON, 
        category_map=category_map, 
        drop_empty=Config.DROP_EMPTY
    )
    
    # 汇总报告
    print(f"\n{'='*60}")
    print(f"数据集统计汇总")
    print(f"{'='*60}")
    print(f"\n训练集:")
    print(f"  • 原始图片数: {train_stats['total_images']}")
    print(f"  • 原始标注数: {train_stats['total_annotations']}")
    print(f"  • 过滤后标注数: {train_stats['filtered_annotations']}")
    print(f"  • 有标注的图片数: {train_stats['images_with_annotations']}")
    print(f"  • 无标注的图片数: {train_stats['images_without_annotations']}")
    print(f"  • 存在的图片文件: {train_stats['existing_image_files']}")
    print(f"  • 缺失的图片文件: {train_stats['missing_image_files']}")
    
    print(f"\n验证集:")
    print(f"  • 原始图片数: {val_stats['total_images']}")
    print(f"  • 原始标注数: {val_stats['total_annotations']}")
    print(f"  • 过滤后标注数: {val_stats['filtered_annotations']}")
    print(f"  • 有标注的图片数: {val_stats['images_with_annotations']}")
    print(f"  • 无标注的图片数: {val_stats['images_without_annotations']}")
    print(f"  • 存在的图片文件: {val_stats['existing_image_files']}")
    print(f"  • 缺失的图片文件: {val_stats['missing_image_files']}")
    
    print(f"\n总计:")
    print(f"  • 总图片数: {train_stats['total_images'] + val_stats['total_images']}")
    print(f"  • 总标注数: {train_stats['total_annotations'] + val_stats['total_annotations']}")
    print(f"  • 总过滤后标注数: {train_stats['filtered_annotations'] + val_stats['filtered_annotations']}")
    print(f"  • 总训练图片数: {train_stats['images_with_annotations']}")
    print(f"  • 总验证图片数: {val_stats['images_with_annotations']}")
    
    # 保存统计结果到文件
    stats = {
        'train': train_stats,
        'val': val_stats,
        'config': {
            'IMAGE_DIR': Config.IMAGE_DIR,
            'TRAIN_JSON': Config.TRAIN_JSON,
            'VAL_JSON': Config.VAL_JSON,
            'SINGLE_CAT_ID': Config.SINGLE_CAT_ID,
            'DROP_EMPTY': Config.DROP_EMPTY,
            'category_map': category_map
        }
    }
    
    with open('dataset_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n统计结果已保存到: dataset_stats.json")
    
    return stats

if __name__ == '__main__':
    count_dataset_stats()