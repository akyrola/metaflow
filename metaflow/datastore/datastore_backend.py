from collections import namedtuple
import re

from .exceptions import DataException


class CloseAfterUse(object):
    """
    Class that can be used to wrap data and a closer (cleanup code).
    This class should be used in a with statement and, when the with
    scope exists, `close` will be called on the closer object
    """
    def __init__(self, data, closer=None):
        self.data = data
        self._closer = closer

    def __enter__(self):
        return self.data

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._closer:
            self._closer.close()


class DataStoreBackend(object):
    """
    A DataStoreBackend defines the interface by which the higher-level
    datastores communicate with the actual storage system

    Both the ContentAddressedStore and the TaskDataStore use these methods to
    read/write/list from the actual storage system. These methods are meant to
    be low-level; they are in a class to provide better abstraction but this
    class itself is not meant to be initialized.
    """
    TYPE = None
    datastore_root = None
    path_rexp = None

    list_content_result = namedtuple('list_content_result', 'path is_file')

    def __init__(self, root=None):
        self.datastore_root = root if root else self.datastore_root

    @classmethod
    def get_datastore_root_from_config(cls, echo, create_on_absent=True):
        """Returns a default choice for datastore_root from metaflow_config

        Parameters
        ----------
        echo : function
            Function to use to print out messages
        create_on_absent : bool, optional
            Create the datastore root if it doesn't exist, by default True
        """
        raise NotImplementedError

    @classmethod
    def get_datastore_root_from_location(cls, path, flow_name):
        """Extracts the datastore_root location from a path using
        a content-addressed store.
        
        NOTE: This leaks some detail of the content-addressed store so not ideal

        This method will raise an exception if the flow_name is not as expected

        Parameters
        ----------
        path : str
            Location from which to extract the datastore root value
        flow_name : str
            Flow name (for verification purposes)

        Returns
        -------
        str
            The datastore_root value that can be used to initialize an instance
            of this datastore backend.

        Raises
        ------
        DataException
            Raised if the path is not a valid path from this datastore.
        """
        if cls.path_rexp is None:
            cls.path_rexp = re.compile(cls.path_join(
                '(?P<root>.*)',
                '(?P<flow_name>[_a-zA-Z][_a-zA-Z0-9]+)',
                'data',
                '(?P<init>[0-9a-f]{2})',
                '(?:r_)?(?P=init)[0-9a-f]{38}'))
        m = cls.path_rexp.match(path)
        if not m or m.group('flow_name') != flow_name:
            raise DataException(
                "Location %s does not correspond to a valid location for flow %s"
                % (path, flow_name))
        return m.group('root')

    @classmethod
    def path_join(cls, *components):
        if len(components) == 0:
            return ''
        component = components[0].rstrip('/')
        components = [component] + [c.strip('/') for c in components[1:]]
        return '/'.join(components)

    @classmethod
    def path_split(cls, path):
        return path.split('/')

    @classmethod
    def basename(cls, path):
        return path.split('/')[-1]

    @classmethod
    def dirname(cls, path):
        return path.rsplit('/', 1)[0]

    def full_uri(self, path):
        return self.path_join(self.datastore_root, path)

    def is_file(self, paths):
        """
        Returns True or False depending on whether path refers to a valid
        file-like object

        This method returns False if path points to a directory

        Parameters
        ----------
        path : List[string]
            Path to the object

        Returns
        -------
        List[bool]
        """
        raise NotImplementedError

    def info_file(self, path):
        """
        Returns a tuple where the first element is True or False depending on
        whether path refers to a valid file-like object (like is_file) and the
        second element is a dictionary of metadata associated with the file or
        None if the file does not exist or there is no metadata.

        Parameters
        ----------
        path : string
            Path to the object

        Returns
        -------
        tuple
            (bool, dict)
        """
        raise NotImplementedError

    def list_content(self, paths):
        """
        Lists the content of the datastore in the directory indicated by 'paths'.

        This is similar to executing a 'ls'; it will only list the content one
        level down and simply returns the paths to the elements present as well
        as whether or not those elements are files (if not, they are further
        directories that can be traversed)

        The path returned always include the path passed in. As an example,
        if your filesystem contains the files: A/b.txt A/c.txt and the directory
        A/D, on return, you would get, for an input of ['A']:
        [('A/b.txt', True), ('A/c.txt', True), ('A/D', False)]

        Parameters
        ----------
        paths : List[string]
            Directories to list

        Returns
        -------
        List[list_content_result]
            Content of the directory
        """
        raise NotImplementedError

    def save_bytes(self, path_and_bytes, overwrite=False):
        """
        Creates objects and stores them in the datastore.

        If overwrite is False, any existing object will not be overwritten and
        an error will be returned.

        The objects are specified in the objects dictionary where the key is the
        path to store the object and the value is a file-like object from which
        bytes can be read.

        Parameters
        ----------
        path_and_bytes : Dict: string ->
                (TransformableObject(RawIOBase or BufferedIOBase), dict)
            Objects to store; the first element in the tuple is the actual data
            to store and the dictionary is additional metadata to store. Keys
            for the metadata must be ascii only string and elements can be
            anything that can be converted to a string using json.dumps. If you
            have no metadata, you can simply pass a RawIOBase or BufferedIOBase.
        overwrite : bool
            True if the objects can be overwritten. Defaults to False.

        Returns
        -------
        None
        """
        raise NotImplementedError

    def load_bytes(self, paths):
        """
        Gets objects from the datastore

        Note that objects may be fetched in parallel so if order is important
        for your consistency model, the caller is responsible for calling this
        multiple times in the proper order.

        Parameters
        ----------
        paths : List[string]
            Paths to fetch

        Returns
        -------
        CloseAfterUse :
            A CloseAfterUse which should be used in a with statement. The data
            in the CloseAfterUse will be a dictionary string -> (path, dict).
            The key is the path fetched and the value is a tuple containing:
              - a path indicating the file that needs to be read to get the object.
                This path may not be valid outside of the CloseAfterUse scope
              - a dictionary containing any additional metadata that was stored
              or None if no metadata was provided.
            If the path could not be loaded, returns None for that path
        """
        raise NotImplementedError