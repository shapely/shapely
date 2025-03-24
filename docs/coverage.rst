Coverage operations
===================

.. currentmodule:: shapely

.. autosummary::
   :toctree: reference/

{% for function in get_module_functions("_coverage") %}
   {{ function }}
{% endfor %}
   coverage_union
   coverage_union_all
