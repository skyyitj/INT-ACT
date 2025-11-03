# 机械臂夹爪6D位姿图像标注

这个项目展示了如何在虚拟环境图像上标注机械臂夹爪的6D位姿信息。

## 功能特性

- 从 SimplerEnv 获取机械臂末端执行器的6D位姿（位置 + 姿态）
- 获取虚拟环境的图像和相机参数
- 将3D位姿投影到2D图像坐标
- 在图像上绘制坐标系和位姿信息标注
- 支持多种标注模式（简化模式和精确模式）

## 文件说明

### 主要文件

1. **`get_robot_pose_demo.py`** - 核心功能实现
   - `get_robot_ee_pose_from_simpler_env()` - 获取机械臂位姿
   - `get_camera_params_from_env()` - 获取相机参数
   - `project_3d_to_2d()` - 3D到2D投影
   - `draw_coordinate_frame()` - 绘制坐标系
   - `draw_pose_annotation()` - 绘制位姿标注
   - `annotate_robot_pose_on_image()` - 主标注函数

2. **`pose_annotation_example.py`** - 使用示例
   - 提供多种演示模式
   - 包含简化和高级标注功能
   - 位姿可视化图表

## 安装要求

```bash
# 安装基础依赖
pip install numpy opencv-python matplotlib

# 安装 SimplerEnv 和相关依赖
# 请参考 SimplerEnv 官方文档进行安装
```

## 使用方法

### 1. 基本使用

```python
from get_robot_pose_demo import annotate_robot_pose_on_image

# 运行位姿标注
annotated_image = annotate_robot_pose_on_image()
```

### 2. 运行示例脚本

```bash
# 运行示例脚本
python pose_annotation_example.py
```

### 3. 分步使用

```python
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from get_robot_pose_demo import (
    get_robot_ee_pose_from_simpler_env,
    get_camera_params_from_env,
    project_3d_to_2d,
    draw_coordinate_frame,
    draw_pose_annotation
)

# 1. 获取机械臂位姿
pose_info = get_robot_ee_pose_from_simpler_env()
env = pose_info['env']
obs = pose_info['obs']

# 2. 获取图像
image = get_image_from_maniskill2_obs_dict(env, obs)

# 3. 获取相机参数
camera_params = get_camera_params_from_env(env)

# 4. 进行3D到2D投影
if camera_params:
    camera_name = list(camera_params.keys())[0]
    params = camera_params[camera_name]
    intrinsic_matrix = params['intrinsic_cv']
    extrinsic_matrix = params['extrinsic_cv']
    
    position_2d = project_3d_to_2d(
        pose_info['position'], 
        intrinsic_matrix, 
        extrinsic_matrix
    )
    
    # 5. 绘制标注
    if position_2d:
        annotated_image = image.copy()
        annotated_image = draw_coordinate_frame(annotated_image, position_2d)
        annotated_image = draw_pose_annotation(
            annotated_image, 
            position_2d, 
            pose_info['quaternion'], 
            pose_info['gripper_width']
        )

# 6. 清理环境
env.close()
```

## 输出结果

运行脚本后会生成以下文件：

1. **`robot_pose_annotation.png`** - 标注了6D位姿的图像
2. **`pose_annotation_comparison.png`** - 原始图像与标注图像的对比
3. **`pose_visualization.png`** - 位姿信息可视化图表
4. **`simple_pose_annotation.png`** - 简化模式标注图像

## 标注内容

### 坐标系标注
- **X轴** - 红色箭头
- **Y轴** - 绿色箭头  
- **Z轴** - 蓝色箭头
- **原点** - 白色圆点

### 位姿信息标注
- **位置坐标** - 夹爪在图像中的2D坐标
- **四元数** - 夹爪的姿态信息 (w, x, y, z)
- **夹爪开合度** - 夹爪的开合状态 (0=闭合, 1=张开)

## 技术细节

### 6D位姿表示
- **位置**: 3D坐标 (x, y, z)，单位：米
- **姿态**: 四元数 (w, x, y, z)，表示旋转
- **夹爪状态**: 开合度 (0-1)，0表示闭合，1表示张开

### 相机投影
使用相机内参和外参矩阵将3D世界坐标投影到2D图像坐标：

```python
# 3D点投影到2D图像坐标
point_3d_homo = [x, y, z, 1]  # 齐次坐标
point_cam = extrinsic_matrix @ point_3d_homo  # 世界坐标 -> 相机坐标
point_2d_homo = intrinsic_matrix @ point_cam[:3]  # 相机坐标 -> 图像坐标
u = point_2d_homo[0] / point_2d_homo[2]  # 像素坐标
v = point_2d_homo[1] / point_2d_homo[2]
```

### 坐标系转换
- **世界坐标系** - 环境中的全局坐标系
- **相机坐标系** - 相机的局部坐标系
- **图像坐标系** - 2D图像像素坐标系

## 故障排除

### 常见问题

1. **无法获取相机参数**
   - 检查 SimplerEnv 是否正确安装
   - 确认环境支持相机参数获取
   - 使用简化模式作为备选方案

2. **3D点投影失败**
   - 夹爪可能不在相机视野内
   - 检查相机参数是否正确
   - 验证3D点坐标的有效性

3. **图像显示问题**
   - 确保安装了 matplotlib
   - 检查图像格式和颜色空间
   - 在支持的环境中运行

### 调试建议

```python
# 启用详细输出
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查相机参数
camera_params = get_camera_params_from_env(env)
if camera_params:
    for name, params in camera_params.items():
        print(f"相机 {name}:")
        print(f"  内参: {params['intrinsic_cv']}")
        print(f"  外参: {params['extrinsic_cv']}")
```

## 扩展功能

### 自定义标注样式
可以修改 `draw_coordinate_frame()` 和 `draw_pose_annotation()` 函数来自定义标注样式：

```python
def custom_draw_pose_annotation(image, position_2d, quaternion, gripper_width):
    # 自定义标注样式
    u, v = position_2d
    
    # 绘制自定义标记
    cv2.circle(image, (u, v), 15, (255, 0, 255), -1)  # 紫色圆点
    
    # 添加自定义文本
    cv2.putText(image, "ROBOT GRIPPER", (u + 20, v), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image
```

### 多相机支持
可以扩展代码以支持多个相机的位姿标注：

```python
def annotate_pose_multiple_cameras(env, obs, pose_info):
    camera_params = get_camera_params_from_env(env)
    annotated_images = {}
    
    for camera_name, params in camera_params.items():
        # 为每个相机生成标注图像
        intrinsic_matrix = params['intrinsic_cv']
        extrinsic_matrix = params['extrinsic_cv']
        
        position_2d = project_3d_to_2d(
            pose_info['position'], 
            intrinsic_matrix, 
            extrinsic_matrix
        )
        
        if position_2d:
            # 获取对应相机的图像
            image = get_camera_image(env, obs, camera_name)
            annotated_image = draw_pose_annotation(image, position_2d, ...)
            annotated_images[camera_name] = annotated_image
    
    return annotated_images
```

## 许可证

请参考项目根目录的许可证文件。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目。

## 联系方式

如有问题，请通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件到项目维护者
