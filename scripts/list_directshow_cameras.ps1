$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;

public static class MediFlowDirectShowCameras {
    [ComImport, Guid("29840822-5B84-11D0-BD3B-00A0C911CE86")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface ICreateDevEnum {
        [PreserveSig] int CreateClassEnumerator(ref Guid category, out IEnumMoniker items, int flags);
    }

    [ComImport, Guid("55272A00-42CB-11CE-8135-00AA004BB851")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IPropertyBag {
        [PreserveSig] int Read([MarshalAs(UnmanagedType.LPWStr)] string key,
            [Out, MarshalAs(UnmanagedType.Struct)] out object value, IntPtr errorLog);
    }

    public static string[] Names() {
        var deviceEnumerator = new Guid("62BE5D10-60EB-11D0-BD3B-00A0C911CE86");
        var videoInputs = new Guid("860BB310-5D01-11D0-BD3B-00A0C911CE86");
        var propertyBag = typeof(IPropertyBag).GUID;
        var enumerator = (ICreateDevEnum)Activator.CreateInstance(Type.GetTypeFromCLSID(deviceEnumerator));
        IEnumMoniker items;
        if (enumerator.CreateClassEnumerator(ref videoInputs, out items, 0) != 0) {
            return new string[0];
        }

        var names = new List<string>();
        var moniker = new IMoniker[1];
        while (items.Next(1, moniker, IntPtr.Zero) == 0) {
            object bag;
            moniker[0].BindToStorage(null, null, ref propertyBag, out bag);
            object name;
            int result = ((IPropertyBag)bag).Read("FriendlyName", out name, IntPtr.Zero);
            names.Add(result == 0 ? name as string : null);
        }
        return names.ToArray();
    }
}
'@

# Preserve the DirectShow ordinal even when a camera has no friendly name.
ConvertTo-Json -InputObject ([MediFlowDirectShowCameras]::Names()) -Compress
