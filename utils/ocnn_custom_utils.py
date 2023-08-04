import torch
import ocnn
from ocnn.octree import Octree, Points

ocnn.dataset.Transform

class CustomTransform:
    r''' [CUSTOMIZED BY ME FOR BIOMASS DL] A boilerplate class which transforms an input data for :obj:`ocnn`.
    The input data is first converted to :class:`Points`, then randomly transformed
    (if enabled), and converted to an :class:`Octree`.

    Args:
      depth (int): The octree depth.
      full_depth (int): The octree layers with a depth small than
          :attr:`full_depth` are forced to be full.
      use_normals (bool): Whether to use surface normals
      orient_normal (bool): Orient point normals along the specified axis, which is
          useful when normals are not oriented.
    '''

    def __init__(self, depth: int,
                 full_depth: int,
                 use_normals=False,
                 orient_normal=False,
                 **kwargs):

        super().__init__()

        # for octree building
        self.use_normals = use_normals
        self.depth = depth
        self.full_depth = full_depth

        # for other transformations
        self.orient_normal = orient_normal

    def __call__(self, sample: dict, idx: int):
        r''''''

        points = self.preprocess(sample, idx)
        output = self.transform(points, idx)
        output['octree'] = self.points2octree(output['points'])
        return output

    def preprocess(self, sample: dict, idx: int):
        r''' Transforms :attr:`sample` to :class:`Points` and performs some specific
        transformations, like normalization.
        '''

        # Select coordinates
        xyz = torch.from_numpy(sample['points']).float()
        # Convert features to tensor (if they are available)
        if sample['features'] is not None:
            features = torch.from_numpy(sample['features']).float()
        else:
            features = None
        # Select normals
        if self.use_normals:
            normals = torch.from_numpy(sample['normals']).float()
        else:
            normals = None

        # Convert to points object that is compatible with octree
        points = Points(xyz, normals=normals, features=features)

        # Need to normalize the point cloud into one unit sphere in [-0.8, 0.8]
        bbmin, bbmax = points.bbox()
        points.normalize(bbmin, bbmax, scale=0.8)
        points.scale(torch.Tensor([0.8, 0.8, 0.8]))

        return points

    def transform(self, points: Points, idx: int):
        r''' Applies the general transformations provided by :obj:`ocnn`.
        '''

        if self.orient_normal:
            points.orient_normal(self.orient_normal)

        # !!! NOTE: Clip the point cloud to [-1, 1] before building the octree
        inbox_mask = points.clip(min=-1, max=1)
        return {'points': points, 'inbox_mask': inbox_mask}

    def points2octree(self, points: Points):
        r''' Converts the input :attr:`points` to an octree.
        '''

        octree = Octree(self.depth, self.full_depth)
        octree.build_octree(points)
        return octree


class CustomCollateBatch:
    r""" Merge a list of octrees and points into a batch.
  """

    def __init__(self, cfg, batch_size: int, merge_points: bool = False):
        self.merge_points = merge_points
        self.batch_size = batch_size
        self.cfg = cfg

    def __call__(self, batch: list):
        assert type(batch) == list

        outputs = {}
        for key in batch[0].keys():
            outputs[key] = [b[key] for b in batch]

            # Merge a batch of octrees into one super octree
            if 'octree' in key:
                octree = ocnn.octree.merge_octrees(outputs[key])
                # NOTE: remember to construct the neighbor indices
                octree.construct_all_neigh()
                outputs[key] = octree

            # Merge a batch of points
            if 'points' in key and self.merge_points:
                outputs[key] = ocnn.octree.merge_points(outputs[key])

            # Convert the labels to a Tensor
            if 'target' in key:
                num_samples = len(outputs['target'])
                if self.cfg['target'] == 'biomass_comps':
                    target_reshape = torch.cat(outputs['target'])
                    num_targets = 4
                elif self.cfg['target'] == "total_agb":
                    target_reshape = torch.stack(outputs['target'])
                    num_targets = 1
                else:
                    raise Exception(f"Target: {self.cfg['target']} is not supported")

                target_reshape = torch.reshape(target_reshape, (num_samples, num_targets))
                outputs['target'] = target_reshape

        return outputs
