"""
This top-level module is organized as follows:

   tools            stand-alone helper functions
   base             basic, generic functionality, mostly in the form of classes, to be used as building blocks for specific applications & projects
   applications     application-specific functionality, tailored towards specific data sources etc...  (e.g. for dealing with data from vedur.is)
   projects         specific projects looking at a given use case & application  (e.g. 1 volcano, based on a specific data source in a given period)

More specific (=more to the bottom of the list) sub-modules can import functionality from more generic (=more to the top of the list) modules, but
  not vice versa.

"""
