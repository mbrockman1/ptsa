# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See the COPYING file distributed along with the PTSA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
from numpy.testing import TestCase,\
    assert_array_equal, assert_array_almost_equal
from numpy.random import random_sample as rnd

from dimarray import DimArray, Dim
from dimarray import AttrArray

import pickle as pickle

# Dim class


class test_Dim(TestCase):
    def setUp(self):
        pass

    def test_new(self):
        # should raise AttributeError if no name is specified:
        self.assertRaises(AttributeError, Dim, list(range(3)))
        # should raise ValueError if not 1-D:
        self.assertRaises(ValueError, Dim, rnd((2, 3)), name='test')
        # should raise ValueError if data is not unique
        self.assertRaises(ValueError, Dim, [1, 2, 2, 3], name='test')
        # should work fine with any number of dimensions as long as it
        # is squeezable or expandable to 1-D:
        tst = Dim(rnd((3, 1, 1, 1, 1)), name='test')
        self.assertEqual(tst.name, 'test')
        tst = Dim(np.array(5), name='test2')
        self.assertEqual(tst.name, 'test2')
        # custom attributes should work, too:
        tst = Dim(list(range(2)), name='test3', custom='attribute')
        self.assertEqual(tst.name, 'test3')
        self.assertEqual(tst.custom, 'attribute')
        # should raise Attribute Error if name is removed:
        self.assertRaises(AttributeError, tst.__setattr__, 'name', None)

    def test_pickle(self):
        # make sure we can pickle this thing
        dat = Dim(np.random.rand(10), name='randvals')

        # dump to string
        pstr = pickle.dumps(dat)

        # load to new variable
        dat2 = pickle.loads(pstr)

        # make sure data same
        assert_array_equal(dat, dat2)

        # make sure has attr and it's correct
        self.assertTrue(hasattr(dat2, '_attrs'))
        self.assertTrue(hasattr(dat2, 'name'))
        self.assertEqual(dat2.name, 'randvals')

        # make sure has required attr
        self.assertTrue(hasattr(dat2, '_required_attrs'))
        self.assertEqual(dat._required_attrs, dat2._required_attrs)


# DimArray class
class test_DimArray(TestCase):
    def setUp(self):
        pass

    def test_new(self):
        # should raise Error if dims contains non-Dim instances:
        self.assertRaises(AttributeError, DimArray, np.random.rand(5, 10),
                          dims=np.arange(4))
        self.assertRaises(AttributeError, DimArray, np.random.rand(5, 10),
                          dims=[Dim(list(range(5)), name='freqs', unit='Hz'),
                                AttrArray(list(range(10)), name='time', unit='sec')])
        self.assertRaises(AttributeError, DimArray, np.random.rand(5, 10),
                          dims=[AttrArray(list(range(5)), name='freqs', unit='Hz'),
                                Dim(list(range(10)), name='time', unit='sec')])

        # should throw Error if dims do not match data shape:
        self.assertRaises(AttributeError, DimArray, np.random.rand(5, 10),
                          dims=[Dim(list(range(10)), name='freqs', unit='Hz'),
                                Dim(list(range(5)), name='time', unit='sec')])
        self.assertRaises(AttributeError, DimArray, np.random.rand(5, 10),
                          dims=[Dim(list(range(5)), name='freqs', unit='Hz')])

        # should throw Error if 2 dims have the same name:
        self.assertRaises(AttributeError, DimArray, np.random.rand(10, 5),
                          dims=[Dim(list(range(10)), name='dim1', unit='Hz'),
                                Dim(list(range(5)), name='dim1', unit='sec')])
        self.assertRaises(AttributeError, DimArray, np.random.rand(10, 3, 5),
                          dims=[Dim(list(range(10)), name='dim1', unit='Hz'),
                                Dim(list(range(3)), name='dim2', unit='Hz'),
                                Dim(list(range(5)), name='dim1', unit='sec')])
        self.assertRaises(AttributeError, DimArray, np.random.rand(10, 3, 5),
                          dims=[Dim(list(range(10)), name='dim1', unit='Hz'),
                                Dim(list(range(3)), name='dim1', unit='Hz'),
                                Dim(list(range(5)), name='dim1', unit='sec')])

        # should throw Error if a dim name is not a valid identifier:
        self.assertRaises(AttributeError, DimArray, np.random.rand(10, 5),
                          dims=[Dim(list(range(10)), name='dim1', unit='Hz'),
                                Dim(list(range(5)), name='dim 2', unit='sec')])
        self.assertRaises(AttributeError, DimArray, np.random.rand(10, 5),
                          dims=[Dim(list(range(10)), name='dim 1', unit='Hz'),
                                Dim(list(range(5)), name='dim2', unit='sec')])
        self.assertRaises(AttributeError, DimArray, np.random.rand(10, 5),
                          dims=[Dim(list(range(10)), name='dim 1', unit='Hz'),
                                Dim(list(range(5)), name='dim 2', unit='sec')])
        self.assertRaises(AttributeError, DimArray, np.random.rand(10, 5),
                          dims=[Dim(list(range(10)), name='dim1', unit='Hz'),
                                Dim(list(range(5)), name='dim$2', unit='sec')])
        self.assertRaises(AttributeError, DimArray, np.random.rand(10, 5),
                          dims=[Dim(list(range(10)), name='$dim1', unit='Hz'),
                                Dim(list(range(5)), name='dim2', unit='sec')])
        self.assertRaises(AttributeError, DimArray, np.random.rand(10, 5),
                          dims=[Dim(list(range(10)), name='1dim1', unit='Hz'),
                                Dim(list(range(5)), name='dim:2', unit='sec')])
        self.assertRaises(AttributeError, DimArray, np.random.rand(10, 5),
                          dims=[Dim(list(range(10)), name='dim1', unit='Hz'),
                                Dim(list(range(5)), name='', unit='sec')])
        self.assertRaises(AttributeError, DimArray, np.random.rand(10, 5),
                          dims=[Dim(list(range(10)), name='', unit='Hz'),
                                Dim(list(range(5)), name='dim2', unit='sec')])
        self.assertRaises(AttributeError, DimArray, np.random.rand(10, 5),
                          dims=[Dim(list(range(10)), name='', unit='Hz'),
                                Dim(list(range(5)), name='', unit='sec')])

        # this is a proper initialization:
        dat = DimArray(np.random.rand(5, 10),
                       dims=[Dim(list(range(5)), name='freqs', unit='Hz'),
                             Dim(list(range(10)), name='time', unit='sec')])
        # should raise Attribute Error if dims is removed:
        self.assertRaises(AttributeError, dat.__setattr__, 'dims', None)
        # ensure dim_names attribute is set properly:
        self.assertEqual(dat.dim_names, ['freqs', 'time'])
        # ensure proper shape
        self.assertEqual(dat.shape, (5, 10))
        # ensure dims have proper lengths:
        self.assertEqual(len(dat.dims[0]), 5)
        self.assertEqual(len(dat.dims[1]), 10)
        # ensure that dims attributes are copied properly:
        self.assertEqual(dat.dims[0].unit, 'Hz')
        self.assertEqual(dat.dims[1].unit, 'sec')
        # check that dims values are preserved:
        self.assertEqual(dat.dims[0][-1], 4)
        self.assertEqual(dat.dims[1][-1], 9)

        dat = DimArray(np.random.rand(2, 4, 5),
                       dims=[Dim(list(range(2)), name='dim1', unit='Hz'),
                             Dim(list(range(4)), name='dim2', bla='bla'),
                             Dim(list(range(5)), name='dim3', attr1='attr1',
                                 attr2='attr2')])
        # ensure dim_names attribute is set properly:
        self.assertEqual(dat.dim_names, ['dim1', 'dim2', 'dim3'])
        # ensure proper shape
        self.assertEqual(dat.shape, (2, 4, 5))
        # ensure dims have proper lengths:
        self.assertEqual(len(dat.dims[0]), 2)
        self.assertEqual(len(dat.dims[1]), 4)
        self.assertEqual(len(dat.dims[2]), 5)
        # ensure that dims attributes are copied properly:
        self.assertEqual(dat.dims[0].unit, 'Hz')
        self.assertEqual(dat.dims[1].bla, 'bla')
        self.assertEqual(dat.dims[2].attr1, 'attr1')
        self.assertEqual(dat.dims[2].attr2, 'attr2')
        # check that dims values are preserved:
        self.assertEqual(dat.dims[0][-1], 1)
        self.assertEqual(dat.dims[1][-1], 3)
        self.assertEqual(dat.dims[2][-1], 4)

        # check filling in of default dims if left out
        dat = DimArray(np.random.rand(4, 3))
        self.assertEqual(dat.dim_names, ['dim1', 'dim2'])
        assert_array_equal(dat['dim1'], np.arange(dat.shape[0]))
        assert_array_equal(dat['dim2'], np.arange(dat.shape[1]))

    def test_pickle(self):
        # make sure we can pickle this thing
        dat = DimArray(np.random.rand(4, 3))

        # dump to string
        pstr = pickle.dumps(dat)

        # load to new variable
        dat2 = pickle.loads(pstr)

        # make sure data same
        assert_array_equal(dat, dat2)

        # make sure has attr and it's correct
        self.assertTrue(hasattr(dat2, '_attrs'))
        self.assertTrue(hasattr(dat2, 'dims'))
        assert_array_equal(dat.dims[0], dat2.dims[0])
        assert_array_equal(dat.dims[1], dat2.dims[1])

        # make sure has required attr
        self.assertTrue(hasattr(dat2, '_required_attrs'))
        self.assertEqual(dat._required_attrs, dat2._required_attrs)

    def test_getitem(self):
        # make ndarray an Dimaray with identical data
        arr = np.random.rand(3)
        dat = DimArray(arr, dims=[Dim(list(range(3)), name='dim1')])
        self.assertEqual(dat[0], dat['dim1==0'])
        self.assertEqual(dat[1], dat['dim1==1'])
        self.assertEqual(dat[2], dat['dim1==2'])
        self.assertEqual(arr[0], dat['dim1==0'])
        self.assertEqual(arr[1], dat['dim1==1'])
        self.assertEqual(arr[2], dat['dim1==2'])

        arr = np.random.rand(3, 2)
        dat = DimArray(arr, dims=[Dim(list(range(3)), name='dim1'),
                                  Dim(list(range(2)), name='dim2')])
        assert_array_equal(dat[:, 0], dat['dim2==0'])
        assert_array_equal(dat[1], dat['dim1==1'])
        assert_array_equal(dat[2], dat['dim1==2'])
        assert_array_equal(arr[0], dat['dim1==0'])
        assert_array_equal(arr[1], dat['dim1==1'])
        assert_array_equal(arr[2], dat['dim1==2'])

        assert_array_equal(dat[0, 0], dat['dim1==0', 'dim2==0'])
        assert_array_equal(dat[0, 1], dat['dim1==0', 'dim2==1'])
        assert_array_equal(dat[1, 0], dat['dim1==1', 'dim2==0'])
        assert_array_equal(dat[1, 1], dat['dim1==1', 'dim2==1'])
        assert_array_equal(dat[2, 0], dat['dim1==2', 'dim2==0'])
        assert_array_equal(dat[2, 1], dat['dim1==2', 'dim2==1'])

        bool_indx = np.zeros(arr.shape, np.bool)
        bool_indx[2, 1] = True
        assert_array_equal(dat[2, 1], dat[bool_indx])
        bool_indx[1, 1] = True
        assert_array_equal(dat[1:3, 1], dat[bool_indx])
        # The below test makes sure that one can work with the results
        # of a Boolean slice into a DimArray object. Because the
        # dimensions get lost with Boolean indices we need to test
        # that there are no complaints from dimension checking (the
        # result should be upcast as an AttrArray):
        test1 = dat[bool_indx] + 1
        test2 = dat[1:3, 1] + 1
        assert_array_equal(test1, test2)

        arr = np.random.rand(3)
        dat = DimArray(arr)
        bool_indx = np.array([True, False, True])
        assert_array_equal(dat[bool_indx], arr[bool_indx])
        assert_array_equal(dat[bool_indx].dims[0], dat.dims[0][bool_indx])

        dat_array = np.random.rand(2, 4, 5)
        dat = DimArray(dat_array,
                       dims=[Dim(list(range(2)), name='dim1', unit='Hz'),
                             Dim(list(range(4)), name='dim2', bla='bla'),
                             Dim(list(range(5)), name='dim3', attr1='attr1',
                                 attr2='attr2')])

        # check that the correct elements are returned:
        self.assertEqual(dat[0, 0, 0], dat_array[0, 0, 0])
        self.assertEqual(dat[0, 1, 2], dat_array[0, 1, 2])
        self.assertEqual(dat[1, 0, 3], dat_array[1, 0, 3])

        # check that the correct elements are returned:
        self.assertEqual(
            dat['dim1==0', 'dim2==0', 'dim3==0'], dat_array[0, 0, 0])
        self.assertEqual(
            dat['dim1==0', 'dim2==1', 'dim3==2'], dat_array[0, 1, 2])
        self.assertEqual(
            dat['dim1==1', 'dim2==0', 'dim3==3'], dat_array[1, 0, 3])

        # check that the returned DimArray and its dims have proper shapes:
        self.assertEqual(dat[0].shape, dat_array[0].shape)
        self.assertEqual(len(dat[0].dims[0]), dat_array[0].shape[0])
        self.assertEqual(len(dat[0].dims[1]), dat_array[0].shape[1])
        self.assertEqual(dat[0].dim_names, ['dim2', 'dim3'])

        self.assertEqual(dat[1].shape, dat_array[1].shape)
        self.assertEqual(len(dat[1].dims[0]), dat_array[1].shape[0])
        self.assertEqual(len(dat[1].dims[1]), dat_array[1].shape[1])
        self.assertEqual(dat[1].dim_names, ['dim2', 'dim3'])

        self.assertEqual(dat[0, 0].shape, dat_array[0, 0].shape)
        self.assertEqual(len(dat[0, 0].dims[0]), dat_array[0, 0].shape[0])
        self.assertEqual(dat[0, 0].dim_names, ['dim3'])

        self.assertEqual(dat[:, :, 0].shape, dat_array[:, :, 0].shape)
        self.assertEqual(len(dat[:, :, 0].dims[0]),
                         dat_array[:, :, 0].shape[0])
        self.assertEqual(len(dat[:, :, 0].dims[1]),
                         dat_array[:, :, 0].shape[1])
        self.assertEqual(dat[:, :, 0].dim_names, ['dim1', 'dim2'])

        self.assertEqual(dat[0:1, 2, 0:3].shape, dat_array[0:1, 2, 0:3].shape)
        self.assertEqual(len(dat[0:1, 2, 0:3].dims[0]),
                         dat_array[0:1, 2, 0:3].shape[0])
        self.assertEqual(len(dat[0:1, 2, 0:3].dims[1]),
                         dat_array[0:1, 2, 0:3].shape[1])
        self.assertEqual(dat[0:1, 2, 0:3].dim_names, ['dim1', 'dim3'])

        self.assertEqual(dat[0:1].shape, dat_array[0:1].shape)
        self.assertEqual(len(dat[0:1].dims[0]),
                         dat_array[0:1].shape[0])
        self.assertEqual(len(dat[0:1].dims[1]),
                         dat_array[0:1].shape[1])
        self.assertEqual(dat[0:1].dim_names, ['dim1', 'dim2', 'dim3'])

        self.assertEqual(dat[1].shape, dat_array[1].shape)
        self.assertEqual(len(dat[1].dims[0]), dat_array[1].shape[0])
        self.assertEqual(len(dat[1].dims[1]), dat_array[1].shape[1])
        self.assertEqual(dat[1].dim_names, ['dim2', 'dim3'])

        self.assertEqual(dat[0, 0].shape, dat_array[0, 0].shape)
        self.assertEqual(len(dat[0, 0].dims[0]), dat_array[0, 0].shape[0])
        self.assertEqual(dat[0, 0].dim_names, ['dim3'])

        self.assertEqual(dat[:, :, 0].shape, dat_array[:, :, 0].shape)
        self.assertEqual(len(dat[:, :, 0].dims[0]),
                         dat_array[:, :, 0].shape[0])
        self.assertEqual(len(dat[:, :, 0].dims[1]),
                         dat_array[:, :, 0].shape[1])
        self.assertEqual(dat[:, :, 0].dim_names, ['dim1', 'dim2'])

        self.assertEqual(dat[0:1, 2, 0:3].shape, dat_array[0:1, 2, 0:3].shape)
        self.assertEqual(len(dat[0:1, 2, 0:3].dims[0]),
                         dat_array[0:1, 2, 0:3].shape[0])
        self.assertEqual(len(dat[0:1, 2, 0:3].dims[1]),
                         dat_array[0:1, 2, 0:3].shape[1])
        self.assertEqual(dat[0:1, 2, 0:3].dim_names, ['dim1', 'dim3'])
        print(dat.dims)
        print(dat['dim2>0'].dims)
        assert_array_equal(dat['dim2>0'].dims[1], dat.dims[1][1:])

        assert_array_equal(dat[1:, 1:], dat['dim1>0', 'dim2>0'])

        # when the name of a Dim instance is given, that dim should be
        # returned:
        self.assertTrue(isinstance(dat['dim1'], Dim))
        self.assertTrue(isinstance(dat['dim2'], Dim))
        self.assertTrue(isinstance(dat['dim3'], Dim))

        self.assertEqual(dat['dim1'].name, 'dim1')
        self.assertEqual(dat['dim1'].unit, 'Hz')
        self.assertEqual(dat['dim1'][-1], 1)
        self.assertEqual(len(dat['dim1']), 2)
        self.assertEqual(dat['dim2'].name, 'dim2')
        self.assertEqual(dat['dim2'].bla, 'bla')
        self.assertEqual(dat['dim2'][-1], 3)
        self.assertEqual(len(dat['dim2']), 4)
        self.assertEqual(dat['dim3'].name, 'dim3')
        self.assertEqual(dat['dim3'].attr1, 'attr1')
        self.assertEqual(dat['dim3'].attr2, 'attr2')
        self.assertEqual(dat['dim3'][-1], 4)
        self.assertEqual(len(dat['dim3']), 5)

        # when another string is given, it should be evaluated:
        self.assertEqual(dat['dim1==0'].shape, (4, 5))
        self.assertEqual(len(dat['dim1==0'].dims[0]), 4)
        self.assertEqual(len(dat['dim1==0'].dims[1]), 5)
        self.assertEqual(dat['dim1==0'].dim_names, ['dim2', 'dim3'])

        self.assertEqual(dat['dim2==1'].shape, (2, 5))
        self.assertEqual(len(dat['dim2==1'].dims[0]), 2)
        self.assertEqual(len(dat['dim2==1'].dims[1]), 5)
        self.assertEqual(dat['dim2==1'].dim_names, ['dim1', 'dim3'])

        self.assertEqual(dat['dim2<2'].shape, (2, 2, 5))
        self.assertEqual(len(dat['dim2<2'].dims[0]), 2)
        self.assertEqual(len(dat['dim2<2'].dims[1]), 2)
        self.assertEqual(len(dat['dim2<2'].dims[2]), 5)
        self.assertEqual(dat['dim2<2'].dim_names, ['dim1', 'dim2', 'dim3'])

        self.assertEqual(dat['dim3!=2'].shape, (2, 4, 4))
        self.assertEqual(len(dat['dim3!=2'].dims[0]), 2)
        self.assertEqual(len(dat['dim3!=2'].dims[1]), 4)
        self.assertEqual(len(dat['dim3!=2'].dims[2]), 4)
        self.assertEqual(dat['dim3!=2'].dim_names, ['dim1', 'dim2', 'dim3'])

        # check that the right values are returned:
        self.assertEqual(dat['dim3!=2'][0, 0, 0], dat_array[0, 0, 0])
        self.assertEqual(dat['dim3!=2'][1, 2, 1], dat_array[1, 2, 1])
        self.assertEqual(dat['dim3!=2'][1, 2, 3], dat_array[1, 2, 4])

        # check indexing with a tuple of arrays and with 1-level dimensions:
        dim1 = Dim(['dim'], 'dim1')
        dim2 = Dim([1, 2], 'dim2')
        dim3 = Dim([3, 4, 5], 'dim3')
        dat = DimArray([[[6, 7, 8], [9, 10, 11]]], [dim1, dim2, dim3])
        self.assertEqual(dat[np.ix_([0], [0, 1], [0, 1])].shape, (1, 2, 2))

        # test string index returning nothing

        # test list index
        dim1 = Dim(['dim'], 'dim1')
        dim2 = Dim([1, 2], 'dim2')
        dim3 = Dim([3, 4, 5], 'dim3')
        dat = DimArray([[[6, 7, 8], [9, 10, 11]]], [dim1, dim2, dim3])
        assert_array_equal(dat[[0]], dat[np.array([0])])

    def test_select(self):
        # check indexing with a tuple of arrays and with 1-level dimensions:
        dim1 = Dim(['dim'], 'dim1')
        dim2 = Dim([1, 2], 'dim2')
        dim3 = Dim([3, 4, 5], 'dim3')
        dat = DimArray([[[6, 7, 8], [9, 10, 11]]], [dim1, dim2, dim3])
        self.assertEqual(dat.select(dim2=dat['dim2'] > 1,
                                    dim3=dat['dim3'] > 3).shape, (1, 1, 2))

    def test_find(self):
        # check indexing with a tuple of arrays and with 1-level dimensions:
        dim1 = Dim(['dim'], 'dim1')
        dim2 = Dim([1, 2], 'dim2')
        dim3 = Dim([3, 4, 5], 'dim3')
        dat = DimArray([[[6, 7, 8], [9, 10, 11]]], [dim1, dim2, dim3])
        indx = dat.find(dim2=dat['dim2'] > 1, dim3=dat['dim3'] > 3)
        assert_array_equal(dat.select(dim2=dat['dim2'] > 1, dim3=dat['dim3'] > 3),
                           dat[indx])

    def test_get_axis(self):
        dat = DimArray(np.random.rand(5, 10, 3),
                       dims=[Dim(list(range(5)), name='one'),
                             Dim(list(range(10)), name='two'),
                             Dim(list(range(3)), name='three')], test='tst')
        self.assertEqual(dat.get_axis(0), 0)
        self.assertEqual(dat.get_axis(1), 1)
        self.assertEqual(dat.get_axis(2), 2)
        self.assertEqual(dat.get_axis('one'), 0)
        self.assertEqual(dat.get_axis('two'), 1)
        self.assertEqual(dat.get_axis('three'), 2)

    def test_reshape(self):
        # make ndarray an Dimaray with identical data
        arr = np.random.rand(5, 12, 3, 1)
        dat = DimArray(arr, dims=[Dim(list(range(5)), name='one'),
                                  Dim(list(range(12)), name='two'),
                                  Dim(list(range(3)), name='three'),
                                  Dim(list(range(1)), name='four')], test='tst')
        newshapes = [(5, 2, 2, 3, 3), (2, 3, 5, 3, 2), (15, 12), (6, 2, 15, 1, 1, 1, 1, 1, 1, 1),
                     180, (1, 1, 1, 180, 1, 1, 1)]
        for newshape in newshapes:
            assert_array_equal(arr.reshape(newshape), dat.reshape(newshape))
            assert_array_equal(np.reshape(arr, newshape),
                               np.reshape(dat, newshape))

    def test_resize(self):
        # make ndarray an Dimaray with identical data
        arr = np.random.rand(5, 12, 3, 1)
        dat = DimArray(arr, dims=[Dim(list(range(5)), name='one'),
                                  Dim(list(range(12)), name='two'),
                                  Dim(list(range(3)), name='three'),
                                  Dim(list(range(1)), name='four')], test='tst')
        self.assertRaises(NotImplementedError, dat.resize, (5, 2, 2, 3, 3))

    def test_newaxis(self):
        # make ndarray an Dimaray with identical data
        arr = np.random.rand(5, 12, 3, 1)
        dat = DimArray(arr, dims=[Dim(list(range(5)), name='one'),
                                  Dim(list(range(12)), name='two'),
                                  Dim(list(range(3)), name='three'),
                                  Dim(list(range(1)), name='four')], test='tst')
        # add a new axis at beginning
        d0 = dat[np.newaxis, :]
        self.assertEqual(d0.dim_names[0], 'newaxis_0')
        self.assertEqual(d0.dim_names[-1], 'four')
        self.assertEqual(len(d0.shape), len(arr.shape) + 1)
        # add a new axis at end
        d0 = dat[:, :, :, :, np.newaxis]
        self.assertEqual(d0.dim_names[-1], 'newaxis_4')
        self.assertEqual(d0.dim_names[0], 'one')
        self.assertEqual(len(d0.shape), len(arr.shape) + 1)
        # add two axes at once
        d0 = dat[np.newaxis, :, :, :, :, np.newaxis]
        self.assertEqual(d0.dim_names[-1], 'newaxis_5')
        self.assertEqual(d0.dim_names[0], 'newaxis_0')
        self.assertEqual(len(d0.shape), len(arr.shape) + 2)
        # make sure the attribute is still there
        d0.test = 'tst'

    def test_add_dim(self):
        # make ndarray an Dimaray with identical data
        arr = np.random.rand(5)
        dat = DimArray(arr, dims=[Dim(list(range(5)), name='one')])
        # make new dim to add
        d = Dim(list(range(10)), name='replicate')
        # add it to the dat
        ndat = dat.add_dim(d)
        # test that it worked
        # verify shape
        self.assertEqual(len(ndat.shape), len(dat.shape) + 1)
        self.assertEqual(ndat.shape[0], 10)
        self.assertEqual(ndat.shape[1], 5)
        # verify contents (a couple random spots)
        assert_array_equal(ndat[4], dat)
        assert_array_equal(ndat[7], dat)
        assert_array_equal(ndat.dims[0], d)
        assert_array_equal(ndat.dims[1], dat.dims[0])

    def test_extend(self):
        """Test the extend method"""
        # make ndarrays and DimArrays with identical data
        arr1 = np.arange(256).reshape((4, 4, 4, 4))
        dat1 = DimArray(arr1, dims=[Dim(np.arange(100, 500, 100), name='one'),
                                    Dim(np.arange(30, 70, 10), name='two'),
                                    Dim(np.arange(4), name='three'),
                                    Dim(np.arange(1000, 1200, 50), name='four')],
                        test='tst')
        arr2 = np.arange(256, 512).reshape((4, 4, 4, 4))
        dat2 = DimArray(arr2, dims=[Dim(np.arange(100, 500, 100), name='one'),
                                    Dim(np.arange(30, 70, 10), name='two'),
                                    Dim(np.arange(4, 8), name='three'),
                                    Dim(np.arange(1000, 1200, 50), name='four')],
                        test='tst')
        # extend
        dat1dat2 = dat1.extend(dat2, 'three')
        arr1arr2 = np.concatenate([arr1, arr2], 2)
        assert_array_equal(dat1dat2, arr1arr2)

        # # test making bins on all dimensions:
        # test1a = dat.make_bins('one',2,np.mean)
        # assert_array_equal(test1a.dims[0],np.array([150,350]))
        # test1b = dat.make_bins(0,2,np.mean,bin_labels='sequential')
        # assert_array_equal(test1b.dims[0],np.array([0,1]))
        # assert_array_equal(test1a,test1b)
        # test2a = dat.make_bins('two',2,np.mean)
        # assert_array_equal(test2a.dims[1],np.array([35,55]))
        # test2b = dat.make_bins(1,2,np.mean,bin_labels=['a','b'])
        # assert_array_equal(test2b.dims[1],np.array(['a','b']))
        # assert_array_equal(test2a,test2b)
        # test3a = dat.make_bins('three',2,np.mean,bin_labels='function')
        # assert_array_equal(test3a.dims[2],np.array([0.5,2.5]))
        # test3b = dat.make_bins(2,2,np.mean)
        # assert_array_equal(test3b.dims[2],np.array([0.5,2.5]))
        # assert_array_equal(test3a,test3b)
        # test4a = dat.make_bins('four',2,np.mean)
        # assert_array_equal(test4a.dims[3],np.array([1025,1125]))
        # test4b = dat.make_bins(3,2,np.mean)
        # assert_array_equal(test4b.dims[3],np.array([1025,1125]))
        # assert_array_equal(test4a,test4b)
        # # test specifiying bins:
        # test4c = dat.make_bins('four',[[1000,1100],[1100,2000]],np.mean)
        # test4d = dat.make_bins(3,[[1000,1100],[1100,2000]],np.mean)
        # assert_array_equal(test4c,test4d)
        # assert_array_equal(test4a,test4d)

        # split = np.split
        # # compare output to reproduced output for ndarray:
        # test1c = np.array(split(arr,2,axis=0)).mean(1)
        # assert_array_equal(test1a,test1c)
        # test2c = np.array(split(arr,2,axis=1)).mean(2).transpose([1,0,2,3])
        # assert_array_equal(test2a,test2c)
        # test3c = np.array(split(arr,2,axis=2)).mean(3).transpose([1,2,0,3])
        # assert_array_equal(test3a,test3c)
        # test4e = np.array(split(arr,2,axis=3)).mean(4).transpose([1,2,3,0])
        # assert_array_equal(test4a,test4e)

        # # compare sequential applications of make_bins to desired output:
        # test12a = test1a.make_bins('two',2,np.mean)
        # assert_array_equal(test1a.dims[0],test12a.dims[0])
        # assert_array_equal(test2a.dims[1],test12a.dims[1])
        # test21a = test2a.make_bins('one',2,np.mean)
        # assert_array_equal(test1a.dims[0],test21a.dims[0])
        # assert_array_equal(test2a.dims[1],test21a.dims[1])
        # assert_array_equal(test12a,test21a)
        # test12b = test1a.make_bins(1,2,np.mean)
        # assert_array_equal(test1a.dims[0],test12b.dims[0])
        # assert_array_equal(test2a.dims[1],test12b.dims[1])
        # test21b = test2a.make_bins(0,2,np.mean)
        # assert_array_equal(test1a.dims[0],test21b.dims[0])
        # assert_array_equal(test2a.dims[1],test21b.dims[1])
        # assert_array_equal(test12b,test21b)

        # # check that attributes are preserved:
        # for a in dat._attrs:
        #     if a == 'dims': continue
        #     self.assertEqual(dat.__getattribute__(a),
        #                      test1a.__getattribute__(a))
        #     self.assertEqual(dat.__getattribute__(a),
        #                      test1b.__getattribute__(a))
        #     self.assertEqual(dat.__getattribute__(a),
        #                      test2a.__getattribute__(a))
        #     self.assertEqual(dat.__getattribute__(a),
        #                      test2b.__getattribute__(a))
        #     self.assertEqual(dat.__getattribute__(a),
        #                      test3a.__getattribute__(a))
        #     self.assertEqual(dat.__getattribute__(a),
        #                      test3b.__getattribute__(a))
        #     self.assertEqual(dat.__getattribute__(a),
        #                      test4a.__getattribute__(a))
        #     self.assertEqual(dat.__getattribute__(a),
        #                      test4b.__getattribute__(a))
        #     self.assertEqual(dat.__getattribute__(a),
        #                      test4d.__getattribute__(a))
        #     self.assertEqual(dat.__getattribute__(a),
        #                      test12a.__getattribute__(a))
        #     self.assertEqual(dat.__getattribute__(a),
        #                      test12b.__getattribute__(a))
        #     self.assertEqual(dat.__getattribute__(a),
        #                      test21a.__getattribute__(a))
        #     self.assertEqual(dat.__getattribute__(a),
        #                      test21b.__getattribute__(a))
        # for d,dn in enumerate(dat.dim_names):
        #     self.assertEquals(test1a.dim_names[d],dn)
        #     self.assertEquals(test1b.dim_names[d],dn)
        #     self.assertEquals(test2a.dim_names[d],dn)
        #     self.assertEquals(test2b.dim_names[d],dn)
        #     self.assertEquals(test3a.dim_names[d],dn)
        #     self.assertEquals(test3b.dim_names[d],dn)
        #     self.assertEquals(test4a.dim_names[d],dn)
        #     self.assertEquals(test4b.dim_names[d],dn)
        #     self.assertEquals(test12a.dim_names[d],dn)
        #     self.assertEquals(test12b.dim_names[d],dn)
        #     self.assertEquals(test21a.dim_names[d],dn)
        #     self.assertEquals(test21b.dim_names[d],dn)

        # # test unequal bins:
        # arr = np.arange(256).reshape((4,16,4))
        # dat = DimArray(arr,dims=[Dim(np.arange(4),name='one'),
        #                          Dim(np.arange(16),name='two'),
        #                          Dim(np.arange(4),name='three')],test='tst')

        # self.assertRaises(ValueError,dat.make_bins,'two',3,np.mean)
        # test5a = dat.make_bins('two',3,np.mean,bin_labels=['1st','2nd','3rd'],
        #                        error_on_nonexact=False)
        # test5b = dat.make_bins(1,[[0,6,'1st'],[6,11,'2nd'],[11,16,'3rd']],
        #                        np.mean)
        # assert_array_equal(test5a,test5b)
        # assert_array_equal(test5a.dims[1],np.array(['1st','2nd','3rd']))
        # assert_array_equal(test5a.dims[1],test5b.dims[1])
        # # check that attributes are preserved:
        # for a in dat._attrs:
        #     if a == 'dims': continue
        #     self.assertEqual(dat.__getattribute__(a),
        #                      test5a.__getattribute__(a))
        #     self.assertEqual(dat.__getattribute__(a),
        #                      test5b.__getattribute__(a))
        # for d,dn in enumerate(dat.dim_names):
        #     self.assertEquals(test5a.dim_names[d],dn)
        #     self.assertEquals(test5b.dim_names[d],dn)

    def test_make_bins(self):
        """Test the make_bins method"""
        # make ndarray and DimArray with identical data
        arr = np.arange(256).reshape((4, 4, 4, 4))
        dat = DimArray(arr, dims=[Dim(np.arange(100, 500, 100), name='one'),
                                  Dim(np.arange(30, 70, 10), name='two'),
                                  Dim(np.arange(4), name='three'),
                                  Dim(np.arange(1000, 1200, 50), name='four')],
                       test='tst')

        # test making bins on all dimensions:
        test1a = dat.make_bins('one', 2, np.mean)
        assert_array_equal(test1a.dims[0], np.array([150, 350]))
        test1b = dat.make_bins(0, 2, np.mean, bin_labels='sequential')
        assert_array_equal(test1b.dims[0], np.array([0, 1]))
        assert_array_equal(test1a, test1b)
        test2a = dat.make_bins('two', 2, np.mean)
        assert_array_equal(test2a.dims[1], np.array([35, 55]))
        test2b = dat.make_bins(1, 2, np.mean, bin_labels=['a', 'b'])
        assert_array_equal(test2b.dims[1], np.array(['a', 'b']))
        assert_array_equal(test2a, test2b)
        test3a = dat.make_bins('three', 2, np.mean, bin_labels='function')
        assert_array_equal(test3a.dims[2], np.array([0.5, 2.5]))
        test3b = dat.make_bins(2, 2, np.mean)
        assert_array_equal(test3b.dims[2], np.array([0.5, 2.5]))
        assert_array_equal(test3a, test3b)
        test4a = dat.make_bins('four', 2, np.mean)
        assert_array_equal(test4a.dims[3], np.array([1025, 1125]))
        test4b = dat.make_bins(3, 2, np.mean)
        assert_array_equal(test4b.dims[3], np.array([1025, 1125]))
        assert_array_equal(test4a, test4b)
        # test specifiying bins:
        test4c = dat.make_bins('four', [[1000, 1100], [1100, 2000]], np.mean)
        test4d = dat.make_bins(3, [[1000, 1100], [1100, 2000]], np.mean)
        assert_array_equal(test4c, test4d)
        assert_array_equal(test4a, test4d)

        split = np.split
        # compare output to reproduced output for ndarray:
        test1c = np.array(split(arr, 2, axis=0)).mean(1)
        assert_array_equal(test1a, test1c)
        test2c = np.array(split(arr, 2, axis=1)).mean(
            2).transpose([1, 0, 2, 3])
        assert_array_equal(test2a, test2c)
        test3c = np.array(split(arr, 2, axis=2)).mean(
            3).transpose([1, 2, 0, 3])
        assert_array_equal(test3a, test3c)
        test4e = np.array(split(arr, 2, axis=3)).mean(
            4).transpose([1, 2, 3, 0])
        assert_array_equal(test4a, test4e)

        # compare sequential applications of make_bins to desired output:
        test12a = test1a.make_bins('two', 2, np.mean)
        assert_array_equal(test1a.dims[0], test12a.dims[0])
        assert_array_equal(test2a.dims[1], test12a.dims[1])
        test21a = test2a.make_bins('one', 2, np.mean)
        assert_array_equal(test1a.dims[0], test21a.dims[0])
        assert_array_equal(test2a.dims[1], test21a.dims[1])
        assert_array_equal(test12a, test21a)
        test12b = test1a.make_bins(1, 2, np.mean)
        assert_array_equal(test1a.dims[0], test12b.dims[0])
        assert_array_equal(test2a.dims[1], test12b.dims[1])
        test21b = test2a.make_bins(0, 2, np.mean)
        assert_array_equal(test1a.dims[0], test21b.dims[0])
        assert_array_equal(test2a.dims[1], test21b.dims[1])
        assert_array_equal(test12b, test21b)

        # check that attributes are preserved:
        for a in dat._attrs:
            if a == 'dims':
                continue
            self.assertEqual(dat.__getattribute__(a),
                             test1a.__getattribute__(a))
            self.assertEqual(dat.__getattribute__(a),
                             test1b.__getattribute__(a))
            self.assertEqual(dat.__getattribute__(a),
                             test2a.__getattribute__(a))
            self.assertEqual(dat.__getattribute__(a),
                             test2b.__getattribute__(a))
            self.assertEqual(dat.__getattribute__(a),
                             test3a.__getattribute__(a))
            self.assertEqual(dat.__getattribute__(a),
                             test3b.__getattribute__(a))
            self.assertEqual(dat.__getattribute__(a),
                             test4a.__getattribute__(a))
            self.assertEqual(dat.__getattribute__(a),
                             test4b.__getattribute__(a))
            self.assertEqual(dat.__getattribute__(a),
                             test4d.__getattribute__(a))
            self.assertEqual(dat.__getattribute__(a),
                             test12a.__getattribute__(a))
            self.assertEqual(dat.__getattribute__(a),
                             test12b.__getattribute__(a))
            self.assertEqual(dat.__getattribute__(a),
                             test21a.__getattribute__(a))
            self.assertEqual(dat.__getattribute__(a),
                             test21b.__getattribute__(a))
        for d, dn in enumerate(dat.dim_names):
            self.assertEqual(test1a.dim_names[d], dn)
            self.assertEqual(test1b.dim_names[d], dn)
            self.assertEqual(test2a.dim_names[d], dn)
            self.assertEqual(test2b.dim_names[d], dn)
            self.assertEqual(test3a.dim_names[d], dn)
            self.assertEqual(test3b.dim_names[d], dn)
            self.assertEqual(test4a.dim_names[d], dn)
            self.assertEqual(test4b.dim_names[d], dn)
            self.assertEqual(test12a.dim_names[d], dn)
            self.assertEqual(test12b.dim_names[d], dn)
            self.assertEqual(test21a.dim_names[d], dn)
            self.assertEqual(test21b.dim_names[d], dn)

        # test unequal bins:
        arr = np.arange(256).reshape((4, 16, 4))
        dat = DimArray(arr, dims=[Dim(np.arange(4), name='one'),
                                  Dim(np.arange(16), name='two'),
                                  Dim(np.arange(4), name='three')], test='tst')

        self.assertRaises(ValueError, dat.make_bins, 'two', 3, np.mean)
        test5a = dat.make_bins('two', 3, np.mean, bin_labels=['1st', '2nd', '3rd'],
                               error_on_nonexact=False)
        test5b = dat.make_bins(1, [[0, 6, '1st'], [6, 11, '2nd'], [11, 16, '3rd']],
                               np.mean)
        assert_array_equal(test5a, test5b)
        assert_array_equal(test5a.dims[1], np.array(['1st', '2nd', '3rd']))
        assert_array_equal(test5a.dims[1], test5b.dims[1])
        # check that attributes are preserved:
        for a in dat._attrs:
            if a == 'dims':
                continue
            self.assertEqual(dat.__getattribute__(a),
                             test5a.__getattribute__(a))
            self.assertEqual(dat.__getattribute__(a),
                             test5b.__getattribute__(a))
        for d, dn in enumerate(dat.dim_names):
            self.assertEqual(test5a.dim_names[d], dn)
            self.assertEqual(test5b.dim_names[d], dn)

    def test_funcs(self):
        """Test the numpy functions"""
        # make ndarray an Dimaray with identical data
        arr = np.random.rand(5, 12, 3, 1)
        dat = DimArray(arr, dims=[Dim(list(range(5)), name='one'),
                                  Dim(list(range(12)), name='two'),
                                  Dim(list(range(3)), name='three'),
                                  Dim(list(range(1)), name='four')], test='tst')

        # these are functions that take an axis argument:
        funcs = [np.mean, np.all, np.any, np.argmax, np.argmin, np.argsort,
                 np.cumprod, np.cumsum, np.max, np.mean, np.min, np.prod,
                 np.ptp, np.std, np.sum, np.var]

        # The axes for the ndarray:
        axes_arr = [None, 0, 1, 2, 3, 0, 1, 2, 3]
        # The axes for the DimArray (we want to test indexing them by
        # number and name):
        axes_dat = [None, 0, 1, 2, 3, 'one', 'two', 'three', 'four']

        # loop through the functions and axes:
        for func in funcs:
            for a in range(len(axes_arr)):
                # apply the function to the ndarray and the DimArray
                arr_func = func(arr, axis=axes_arr[a])
                dat_func = func(dat, axis=axes_dat[a])
                # make sure they are the same:
                assert_array_equal(arr_func, dat_func)
                if not(axes_dat[a] is None):
                    # ensure we still have a DimArray
                    self.assertTrue(isinstance(dat_func, DimArray))
                    # ensure that the attributes are preserved
                    self.assertEqual(dat_func.test, 'tst')

        # same tests as above but this time calling the DimArray
        # methods directly (this test is necessary because it is in
        # principle possible for the numpy function to work and the
        # DimArray method not to work (or vice versa):
        for a in range(len(axes_arr)):
            assert_array_equal(arr.all(axes_arr[a]),
                               dat.all(axes_dat[a]))
            assert_array_equal(arr.any(axes_arr[a]),
                               dat.any(axes_dat[a]))
            assert_array_equal(arr.argmax(axes_arr[a]),
                               dat.argmax(axes_dat[a]))
            assert_array_equal(arr.argmin(axes_arr[a]),
                               dat.argmin(axes_dat[a]))
            assert_array_equal(arr.argsort(axes_arr[a]),
                               dat.argsort(axes_dat[a]))
            assert_array_equal(arr.cumprod(axes_arr[a]),
                               dat.cumprod(axes_dat[a]))
            assert_array_equal(arr.cumsum(axes_arr[a]),
                               dat.cumsum(axes_dat[a]))
            assert_array_equal(arr.max(axes_arr[a]),
                               dat.max(axes_dat[a]))
            assert_array_equal(arr.mean(axes_arr[a]),
                               dat.mean(axes_dat[a]))
            assert_array_equal(arr.min(axes_arr[a]),
                               dat.min(axes_dat[a]))
            assert_array_equal(arr.prod(axes_arr[a]),
                               dat.prod(axes_dat[a]))
            assert_array_equal(arr.ptp(axes_arr[a]),
                               dat.ptp(axes_dat[a]))
            assert_array_equal(arr.std(axes_arr[a]),
                               dat.std(axes_dat[a]))
            assert_array_equal(arr.sum(axes_arr[a]),
                               dat.sum(axes_dat[a]))
            assert_array_equal(arr.var(axes_arr[a]),
                               dat.var(axes_dat[a]))
            if not(axes_dat[a] is None):
                self.assertTrue(isinstance(dat.all(axes_arr[a]), DimArray))
                self.assertTrue(isinstance(dat.any(axes_arr[a]), DimArray))
                self.assertTrue(isinstance(dat.argmax(axes_arr[a]), DimArray))
                self.assertTrue(isinstance(dat.argmin(axes_arr[a]), DimArray))
                self.assertTrue(isinstance(dat.argsort(axes_arr[a]), DimArray))
                self.assertTrue(isinstance(dat.cumprod(axes_arr[a]), DimArray))
                self.assertTrue(isinstance(dat.cumsum(axes_arr[a]), DimArray))
                self.assertTrue(isinstance(dat.max(axes_arr[a]), DimArray))
                self.assertTrue(isinstance(dat.mean(axes_arr[a]), DimArray))
                self.assertTrue(isinstance(dat.min(axes_arr[a]), DimArray))
                self.assertTrue(isinstance(dat.prod(axes_arr[a]), DimArray))
                self.assertTrue(isinstance(dat.ptp(axes_arr[a]), DimArray))
                self.assertTrue(isinstance(dat.std(axes_arr[a]), DimArray))
                self.assertTrue(isinstance(dat.sum(axes_arr[a]), DimArray))
                self.assertTrue(isinstance(dat.var(axes_arr[a]), DimArray))

                self.assertEqual(dat.all(axes_arr[a]).test, 'tst')
                self.assertEqual(dat.any(axes_arr[a]).test, 'tst')
                self.assertEqual(dat.argmax(axes_arr[a]).test, 'tst')
                self.assertEqual(dat.argmin(axes_arr[a]).test, 'tst')
                self.assertEqual(dat.argsort(axes_arr[a]).test, 'tst')
                self.assertEqual(dat.cumprod(axes_arr[a]).test, 'tst')
                self.assertEqual(dat.cumsum(axes_arr[a]).test, 'tst')
                self.assertEqual(dat.max(axes_arr[a]).test, 'tst')
                self.assertEqual(dat.mean(axes_arr[a]).test, 'tst')
                self.assertEqual(dat.min(axes_arr[a]).test, 'tst')
                self.assertEqual(dat.prod(axes_arr[a]).test, 'tst')
                self.assertEqual(dat.ptp(axes_arr[a]).test, 'tst')
                self.assertEqual(dat.std(axes_arr[a]).test, 'tst')
                self.assertEqual(dat.sum(axes_arr[a]).test, 'tst')
                self.assertEqual(dat.var(axes_arr[a]).test, 'tst')

        # test functions that require function specific input:
        for a in range(len(axes_arr)):
            if axes_arr[a] is None:
                length = len(arr)
            else:
                length = np.shape(arr)[axes_arr[a]]
            cond = np.random.random(length) > 0.5
            # calling the compress method directly:
            arr_func = arr.compress(cond, axis=axes_arr[a])
            dat_func = dat.compress(cond, axis=axes_dat[a])
            assert_array_equal(arr_func, dat_func)
            if axes_dat[a] is not None:
                self.assertTrue(isinstance(dat_func, DimArray))
                self.assertEqual(dat_func.test, 'tst')
            # calling the numpy compress function:
            arr_func = np.compress(cond, arr, axis=axes_arr[a])
            dat_func = np.compress(cond, dat, axis=axes_dat[a])
            assert_array_equal(arr_func, dat_func)
            if axes_dat[a] is not None:
                self.assertTrue(isinstance(dat_func, DimArray))
                self.assertEqual(dat_func.test, 'tst')

            # the below tests should not run with axis==None:
            if axes_arr[a] is None:
                continue

            reps = np.random.random_integers(low=1, high=10, size=length)
            # calling the repeat method directly:
            arr_func = arr.repeat(reps, axis=axes_arr[a])
            dat_func = dat.repeat(reps, axis=axes_dat[a])
            assert_array_equal(arr_func, dat_func)
            if axes_dat[a] is not None:
                self.assertTrue(isinstance(dat_func, AttrArray))
                self.assertEqual(dat_func.test, 'tst')
            # calling the numpy repeat function:
            arr_func = np.repeat(arr, reps, axis=axes_arr[a])
            dat_func = np.repeat(dat, reps, axis=axes_dat[a])
            assert_array_equal(arr_func, dat_func)
            if axes_dat[a] is not None:
                self.assertTrue(isinstance(dat_func, AttrArray))
                self.assertEqual(dat_func.test, 'tst')

            # skip the last dimension for this test for
            # convenience (the last dimension only has 1 level):
            if a >= 3:
                continue
            indcs = np.arange(len(arr.shape))
            # calling the take method directly (squeeze, to get rid of
            # the last dimension):
            arr_func = arr.squeeze().take(indcs, axis=axes_arr[a])
            dat_func = dat.squeeze().take(indcs, axis=axes_dat[a])
            assert_array_equal(arr_func, dat_func)
            if axes_dat[a]:
                self.assertTrue(isinstance(dat_func, AttrArray))
                self.assertEqual(dat_func.test, 'tst')
            # calling the numpy take function directly (squeeze, to get rid of
            # the last dimension):
            arr_func = np.take(arr.squeeze(), indcs, axis=axes_arr[a])
            dat_func = np.take(dat.squeeze(), indcs, axis=axes_dat[a])
            assert_array_equal(arr_func, dat_func)
            if axes_dat[a]:
                self.assertTrue(isinstance(dat_func, AttrArray))
                self.assertEqual(dat_func.test, 'tst')

        # This should work with numpy 1.2 but doesn't
        # with 1.1.1 or below (therfore commented out for now):
        # arr_func = arr.clip(0.4,0.6)
        # dat_func = dat.clip(0.4,0.6)
        # assert_array_equal(arr_func,dat_func)
        # self.assertTrue(isinstance(dat_func,DimArray))
        # self.assertEquals(dat_func.test,'tst')
        #arr_func = np.clip(arr,0.4,0.6)
        #dat_func = np.clip(dat,0.4,0.6)
        # assert_array_equal(arr_func,dat_func)
        # self.assertTrue(isinstance(dat_func,DimArray))
        # self.assertEquals(dat_func.test,'tst')

        # other functions that don't necessarily take return a
        # DimArray:
        funcs = [np.diagonal, np.nonzero, np.ravel, np.squeeze,
                 np.sort, np.trace, np.transpose]
        for func in funcs:
            arr_func = func(arr)
            dat_func = func(dat)
            assert_array_equal(arr_func, dat_func)

        # same tests as above, but calling the methods directly:
        assert_array_equal(arr.diagonal(), dat.diagonal())
        assert_array_equal(arr.nonzero(), dat.nonzero())
        assert_array_equal(arr.ravel(), dat.ravel())
        assert_array_equal(arr.squeeze(), dat.squeeze())
        assert_array_equal(arr.sort(), dat.sort())
        assert_array_equal(arr.trace(), dat.trace())
        assert_array_equal(arr.transpose(), dat.transpose())
        # there is no numpy.flatten() function, so we only call the
        # method directly:
        assert_array_equal(arr.flatten(), dat.flatten())

        assert_array_equal(arr.swapaxes(0, 1), dat.swapaxes(0, 1))
        self.assertTrue(isinstance(dat.swapaxes(0, 1), DimArray))
        self.assertEqual(dat.swapaxes(0, 1).test, 'tst')
        assert_array_equal(arr.swapaxes(0, 1), dat.swapaxes('one', 'two'))
        self.assertTrue(isinstance(dat.swapaxes('one', 'two'), DimArray))
        self.assertEqual(dat.swapaxes('one', 'two').test, 'tst')
        assert_array_equal(arr.swapaxes(1, 3), dat.swapaxes(1, 3))
        self.assertTrue(isinstance(dat.swapaxes(1, 3), DimArray))
        self.assertEqual(dat.swapaxes(1, 3).test, 'tst')
        assert_array_equal(arr.swapaxes(1, 3), dat.swapaxes('two', 'four'))
        self.assertTrue(isinstance(dat.swapaxes('two', 'four'), DimArray))
        self.assertEqual(dat.swapaxes('two', 'four').test, 'tst')

    def test_ufuncs(self):
        """Test the numpy u-functions"""
        # make ndarray an Dimaray with identical data
        arr = np.random.rand(5)
        dat = DimArray(arr, dims=[Dim(list(range(5)), name='one')],
                       test='tst')

        x = np.ones(10) * dat[[0]]
        self.assertTrue(isinstance(x, np.ndarray))

        x = np.ones(1) * dat[[0]]
        self.assertTrue(isinstance(x, DimArray))

        x = 22 * dat
        self.assertTrue(isinstance(x, DimArray))
        assert_array_equal(x, arr * 22)
