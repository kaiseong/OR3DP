#!/usr/bin/env python3
"""
Point Cloud Visualizer Class
재사용 가능한 Open3D 기반 포인트 클라우드 시각화 컴포넌트
"""
import numpy as np
import open3d as o3d


class PointCloudVisualizer:
    """
    단일 포인트 클라우드 시각화 창 관리 클래스

    Features:
    - 독립적인 시각화 창 생성 및 관리
    - 좌표축 표시/숨김 기능
    - 포인트 클라우드 업데이트
    - 카메라 뷰 설정
    """

    def __init__(
        self,
        window_name="Point Cloud",
        width=640,
        height=480,
        left=0,
        top=0,
        point_size=2.0,
        background_color=(1.0, 1.0, 1.0),
        show_coordinate=False,
        coordinate_size=0.2,
        coordinate_origin=(0.0, 0.0, 0.0)
    ):
        """
        Args:
            window_name: 창 이름
            width: 창 너비
            height: 창 높이
            left: 창 X 위치
            top: 창 Y 위치
            point_size: 포인트 크기
            background_color: 배경색 (R, G, B) in [0, 1]
            show_coordinate: 좌표축 표시 여부
            coordinate_size: 좌표축 크기 (meter)
            coordinate_origin: 좌표축 원점 위치 (x, y, z)
        """
        self.window_name = window_name
        self.show_coordinate = show_coordinate
        self.coordinate_size = coordinate_size
        self.coordinate_origin = coordinate_origin

        # Open3D Visualizer 생성
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name=window_name,
            width=width,
            height=height,
            left=left,
            top=top
        )

        # Render options
        opt = self.vis.get_render_option()
        opt.point_size = point_size
        opt.background_color = np.array(background_color)

        # Point cloud geometry
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)

        # Coordinate frame geometry
        self.coordinate_frame = None
        if show_coordinate:
            self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=coordinate_size,
                origin=coordinate_origin
            )
            self.vis.add_geometry(self.coordinate_frame)

        # View control
        self.view_control = self.vis.get_view_control()

        # State
        self.is_first_update = True

    def update(self, points, colors=None):
        """
        포인트 클라우드 업데이트

        Args:
            points: (N, 3) numpy array, XYZ coordinates
            colors: (N, 3) numpy array, RGB in [0, 1]. None이면 회색
        """
        if len(points) == 0:
            # 빈 포인트 클라우드
            self.pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            self.pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        else:
            self.pcd.points = o3d.utility.Vector3dVector(points)

            if colors is None:
                # Default: 회색
                colors = np.ones((len(points), 3)) * 0.5

            self.pcd.colors = o3d.utility.Vector3dVector(colors)

        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def set_view(self, lookat=None, front=None, up=None, zoom=None):
        """
        카메라 뷰 설정

        Args:
            lookat: [x, y, z] 카메라가 바라보는 점
            front: [x, y, z] 카메라 전방 벡터
            up: [x, y, z] 카메라 상향 벡터
            zoom: float 줌 레벨
        """
        if lookat is not None:
            self.view_control.set_lookat(lookat)

        if front is not None:
            self.view_control.set_front(front)

        if up is not None:
            self.view_control.set_up(up)

        if zoom is not None:
            self.view_control.set_zoom(zoom)

    def toggle_coordinate(self):
        """좌표축 표시/숨김 토글"""
        if self.coordinate_frame is None:
            # 좌표축 생성 및 추가
            self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=self.coordinate_size,
                origin=self.coordinate_origin
            )
            self.vis.add_geometry(self.coordinate_frame)
            self.show_coordinate = True
        else:
            # 좌표축 제거
            self.vis.remove_geometry(self.coordinate_frame)
            self.coordinate_frame = None
            self.show_coordinate = False

    def poll_events(self):
        """
        이벤트 폴링 (창이 닫혔는지 확인)

        Returns:
            bool: 창이 열려있으면 True, 닫혔으면 False
        """
        return self.vis.poll_events()

    def destroy(self):
        """시각화 창 종료"""
        self.vis.destroy_window()
