import os
import cv2
import sys
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.detector import YoloPoseDetector
from core.pose_analyzer import FallAnalyzer

def test_image_processing():
    """测试图片处理功能"""
    print("=== 测试图片处理功能 ===")
    
    # 初始化检测器和分析器
    detector = YoloPoseDetector("yolo11n-pose.pt")
    analyzer = FallAnalyzer(detector)
    
    # 选择一张测试图片
    test_image_path = os.path.join("datasets", "images", "val", "fall001.jpg")
    if not os.path.exists(test_image_path):
        print(f"测试图片不存在: {test_image_path}")
        return False
    
    print(f"正在处理图片: {test_image_path}")
    
    # 读取图片
    image = cv2.imread(test_image_path)
    if image is None:
        print("无法读取图片")
        return False
    
    try:
        # 检测人体姿态
        detections = detector.detect(image)
        print(f"检测到的人体数量: {len(detections)}")
        
        if detections:
            # 分析姿态
            results = analyzer.analyze_multi_poses(detections)
            print(f"分析结果数量: {len(results)}")
            
            # 打印详细结果
            for i, result in enumerate(results):
                print(f"\n人员 {i+1}:")
                print(f"  置信度: {result['confidence']:.2f}")
                print(f"  是否摔倒: {'是' if result['analysis']['is_fall'] else '否'}")
                print(f"  分析置信度: {result['analysis']['confidence']:.2f}")
                print("  特征值:")
                for key, value in result['analysis']['features'].items():
                    # 确保值是Python原生类型
                    if hasattr(value, 'tolist'):
                        value = value.tolist()
                    elif hasattr(value, 'item'):
                        value = value.item()
                    
                    # 根据值的类型选择适当的格式化方式
                    if isinstance(value, (int, float)):
                        print(f"    {key}: {value:.2f}")
                    elif isinstance(value, list):
                        # 处理列表类型（如body_center）
                        formatted_list = [f"{v:.2f}" for v in value]
                        print(f"    {key}: [{', '.join(formatted_list)}]")
                    else:
                        print(f"    {key}: {value}")
            
            # 测试红色文字显示和声音报警功能
            # 模拟一个摔倒检测结果
            import winsound
            print("\n=== 测试报警功能 ===")
            print("检测到摔倒！")
            winsound.Beep(1000, 500)  # 1000Hz频率，500ms持续时间
            
            return True
        else:
            print("未检测到人体")
            return False
            
    except Exception as e:
        print(f"处理图片时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_image_processing()
    if success:
        print("\n✅ 图片处理测试成功！")
    else:
        print("\n❌ 图片处理测试失败！")
        sys.exit(1)