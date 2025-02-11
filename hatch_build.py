"""
Takes responsibility for adding compatible version pinning to the built packages
"""

# pylint: skip-file

import pathlib
from typing import Optional

import tomlkit
from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from hatchling.metadata.plugin.interface import MetadataHookInterface


class PinVersionsMetadataHook(MetadataHookInterface):
    """
    Update the dependency metadata that `hatch` uses to set the dependencies of wheel artefacts.

    This is invoked by `hatch` when `hatch build` is run as part of constructing the package metadata
     before it moves on to executing the next appropriate build step, be it sdist or wheels.
    """

    def update(self, metadata):
        change_count = 0
        # Overlay the pinned dependencies, replacing dependencies if they already exist.
        for pinned_dep in self.config["config"]["pinned_dependencies"]:
            pinned_dep_name = pinned_dep.split(" ")[0]
            for dep in metadata["dependencies"]:
                dep_name = dep.split(" ")[0]
                # If the dependency is already in the list, replace it with the pinned version.
                if dep_name == pinned_dep_name:
                    index = metadata["dependencies"].index(dep)
                    metadata["dependencies"][index] = pinned_dep
                    change_count += 1
                    break
        print(
            f"Updated {change_count} dependencies in hatch's internal dependency metadata"
            f" to pinned versions from config."
        )


class PinVersionsBuildHook(BuildHookInterface):
    """
    Temporarily edit the dependencies in the pyproject.toml file to use the desired pinned versions
     when using `hatch` to build a package for release.

    When `hatch` performs a build in response to running the `hatch build` build command, its plugin system
     will call the initialize method of this hook before the build process starts. This allows the hook to
     update the dependencies in the pyproject.toml file before the build starts. The finalize method is called
     after the build has completed to restore the original dependencies.
    """

    PLUGIN_NAME = "pin-during-build"

    original_dependencies: Optional[list] = None
    made_changes: bool = False

    def initialize(self, version, build_data):
        # To avoid affecting the wheel METADATA file, we only run this hook's logic when
        # the build system is building a sdist artefact.

        print()
        if self.build_config.builder.PLUGIN_NAME != "sdist":
            print("Building wheel artefact. ")
            print("This uses hatch's internal dependency metadata.")
            print()
            return
        else:
            print("Building sdist artefact.")
            print("This does not use hatch's internal dependency metadata.")
            print()
            print("Updating pyproject.toml to contain the correct versions of pinned dependencies.")

        # Get the pinned dependencies, this is a list of package specifier strings that are the dependencies to pin.
        pinned_dependencies = self.metadata.hatch.metadata.hook_config["custom"]["config"]["pinned_dependencies"]

        # Load the toml file
        pyproject_file = pathlib.Path("pyproject.toml")
        toml_data = tomlkit.loads(pyproject_file.read_text())

        # Get the dependencies from the toml file, this is a list of package specifier strings.
        dependencies = toml_data["project"]["dependencies"]

        # Save the original dependencies for later.
        self.original_dependencies = dependencies.copy()

        change_count = 0

        # Update dependencies with pinned versions if they are in the configuration and
        # only if the un-pinned version is present.
        for pinned_dep in pinned_dependencies:
            pinned_dep_name = pinned_dep.split(" ")[0]
            for dep in dependencies:
                dep_name = dep.split(" ")[0]
                # If the dependency is already in the list, replace it with the pinned version.
                if dep_name == pinned_dep_name:
                    index = dependencies.index(dep)
                    dependencies[index] = pinned_dep
                    change_count += 1
                    break

        # If we made any changes, write the changes back to the pyroject.toml file for the build.
        if change_count > 0:
            # Write the changes back to the file.
            pyproject_file.write_text(tomlkit.dumps(toml_data))
            print(
                f"Updated {change_count} dependencies to pinned versions in the pyproject.toml "
                f"which will be incorporated into the final sdist artefact."
            )
            # Set a flag to restore the original dependencies after the build.
            self.made_changes = True
        else:
            print("No dependencies were changed to pinned version.")

        # Update the build data with the pinned dependencies so it populates the METADATA file in the wheel artefact.
        build_data["dependencies"] = dependencies

        return super().initialize(version, build_data)

    def finalize(self, version, build_data, artefact_path):
        # If we have made changes restore the original dependencies after the build to keep the git repo clean.
        if self.made_changes:
            pyproject_file = pathlib.Path("pyproject.toml")
            toml_data = tomlkit.loads(pyproject_file.read_text())
            toml_data["project"]["dependencies"] = self.original_dependencies
            pyproject_file.write_text(tomlkit.dumps(toml_data))
            print()
            print("Build of sdist artefact completed.")
            print("Restored original dependencies in pyproject.toml file.")
            print("This puts the git repo back to its original state.")
            print()
