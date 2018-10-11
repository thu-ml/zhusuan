class ObservationStorage(object):

    def __init__(self):
        pass

    def get_node(self, name, tag=None, shape=None, n_samples=None):
        raise NotImplementedError()

    def merge_with(self, another_storage):
        return StorageCollection([self, another_storage])


class FixedObservations(ObservationStorage):

    def __init__(self, node_dict):
        self._dict = node_dict

    def get_node(self, name, tag=None, shape=None, n_samples=None):
        # TODO optionally, validate the fetched node
        return self._dict.get(name, None)


class BayesianNetStorage(ObservationStorage):

    def __init__(self, bn):
        self._bn = bn

    def get_node(self, name, tag=None, shape=None, n_samples=None):
        if not self._bn.has_node(name, tag):
            return None
        return self._bn.get_node(name, tag=tag, shape=shape, n_samples=n_samples)


class FilteredStorage(ObservationStorage):
    
    def __init__(self, storage, filter_fn):
        self._storage = storage
        self._filter_fn = filter_fn

    def get_node(self, name, tag=None, shape=None, n_samples=None):
        ret = self._storage.get_node(name, tag, shape, n_samples)
        if ret is None or not self._filter_fn(ret):
            return None
        return ret


class StorageCollection(ObservationStorage):

    def __init__(self, storages):
        self._storages = storages

    def get_node(self, name, tag=None, shape=None, n_samples=None):
        ret = [s.get_node(name, tag, shape, n_samples) for s in self._storages]
        ret = [nd for nd in ret if nd is not None]
        assert len(ret) <= 1
        return ret[0] if len(ret) > 0 else None

